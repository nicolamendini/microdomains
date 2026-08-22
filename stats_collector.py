import hashlib
import importlib.metadata
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Image as IPImage
from helpers.wiring_efficiency_utils import *
from neuralsheet import *
import os
import gc
import time
from wakepy import keep

DEVICE = 'cuda'
AGGRESSIVE_CLEANUP = False
DEFAULT_BASE_SEED = 20260728
STRICT_DETERMINISM = True
CUBLAS_WORKSPACE_CONFIG = ':4096:8'

_DATASET_FINGERPRINT_CACHE = {}
_EVALUATION_CACHE = {}
_NULL_BASELINE_CACHE = {}


def _validate_n_reps(n_reps):
    n_reps = int(n_reps)
    if n_reps < 1:
        raise ValueError('n_reps must be at least 1.')
    return n_reps


def _derive_seed(base_seed, *parts):
    """Derive a stable uint32 seed without depending on Python's hash()."""
    payload = '|'.join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little') % (2**32)


def _configure_reproducibility(strict_determinism):
    """Configure deterministic PyTorch execution before CUDA is first used."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', CUBLAS_WORKSPACE_CONFIG)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(strict_determinism)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(
        bool(strict_determinism), warn_only=False)


def _seed_everything(seed):
    """Seed every RNG used directly by the collectors."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_value(*args):
    try:
        result = subprocess.run(
            ['git', *args],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes():
    standalone_source = os.environ.get('NEURALSHEET_STANDALONE_SOURCE')
    if standalone_source:
        path = Path(standalone_source).resolve()
        return {path.name: _sha256_file(path)}

    repository_root = Path(__file__).resolve().parent
    source_paths = [
        repository_root / 'stats_collector.py',
        repository_root / 'run_accdim_and_noise_sweeps.py',
        repository_root / 'neuralsheet.py',
        repository_root / 'nn_template.py',
        repository_root / 'helpers' / 'wiring_efficiency_utils.py',
    ]
    return {
        str(path.relative_to(repository_root)): _sha256_file(path)
        for path in source_paths if path.is_file()
    }


def _dataset_fingerprint(root_dir):
    """Hash the ordered filenames and bytes of the complete input dataset."""
    root_path = Path(root_dir).resolve()
    cache_key = str(root_path)
    if cache_key in _DATASET_FINGERPRINT_CACHE:
        return dict(_DATASET_FINGERPRINT_CACHE[cache_key])

    paths = sorted(path for path in root_path.iterdir() if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    print(
        'fingerprinting input dataset for reproducibility: '
        + str(len(paths))
        + ' files'
    )
    for path in paths:
        relative_name = path.relative_to(root_path).as_posix()
        file_size = path.stat().st_size
        total_bytes += file_size
        digest.update(relative_name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(str(file_size).encode('ascii'))
        digest.update(b'\0')
        with open(path, 'rb') as file_handle:
            for chunk in iter(
                    lambda: file_handle.read(1024 * 1024), b''):
                digest.update(chunk)

    fingerprint = {
        'algorithm': 'sha256_ordered_relative_path_size_and_content',
        'sha256': digest.hexdigest(),
        'file_count': len(paths),
        'total_bytes': total_bytes,
    }
    _DATASET_FINGERPRINT_CACHE[cache_key] = fingerprint
    print('input dataset sha256: ' + fingerprint['sha256'])
    return dict(fingerprint)


def _package_versions():
    packages = [
        'numpy',
        'pillow',
        'scipy',
        'scikit-learn',
        'torch',
        'torchvision',
        'matplotlib',
        'tqdm',
        'umap-learn',
        'wakepy',
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _reproducibility_context(
        root_dir, base_seed, strict_determinism):
    repository_status = _git_value('status', '--porcelain')
    cuda_device = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        cuda_device = {
            'name': properties.name,
            'total_memory_bytes': properties.total_memory,
            'compute_capability': [
                properties.major,
                properties.minor,
            ],
        }

    return {
        'base_seed': int(base_seed),
        'seed_derivation': (
            'uint32 little-endian first 8 bytes of '
            'SHA256(base_seed|component|indices)'
        ),
        'strict_determinism': bool(strict_determinism),
        'deterministic_algorithms_enabled': (
            torch.are_deterministic_algorithms_enabled()
        ),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'environment': {
            name: os.environ.get(name)
            for name in [
                'CUBLAS_WORKSPACE_CONFIG',
                'PYTHONHASHSEED',
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS',
            ]
        },
        'python_version': sys.version,
        'platform': platform.platform(),
        'hostname': platform.node(),
        'packages': _package_versions(),
        'torch_cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cuda_device': cuda_device,
        'git_commit': _git_value('rev-parse', 'HEAD'),
        'git_dirty': bool(repository_status),
        'git_status_porcelain': repository_status,
        'source_sha256': _source_hashes(),
        'input_dataset': _dataset_fingerprint(root_dir),
        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def _run_model(model, image, **kwargs):
    """Run the current L4 collector path without requiring L2/3 parameters."""
    return model(image, layer_3=False, **kwargs)


def _cat_or_none(tensors):
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _require_codes(code_tracker, context):
    if not code_tracker:
        raise RuntimeError(f'No valid responses collected during {context}.')
    return torch.cat(code_tracker, dim=0)


def _as_list(values):
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    return list(values)


def _cleanup(force=False):
    if not (AGGRESSIVE_CLEANUP or force):
        return

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fixed_train_and_evaluation_data(
        root_dir,
        crop_size,
        batch_size,
        num_workers,
        base_seed,
        evaluation_samples=10_000):
    """Create one deterministic set of post-training test presentations."""
    cache_key = (
        str(Path(root_dir).resolve()), int(crop_size), int(batch_size),
        int(num_workers), int(base_seed), int(evaluation_samples),
    )
    if cache_key in _EVALUATION_CACHE:
        return _EVALUATION_CACHE[cache_key]

    full_dataset = RandomCropDataset(root_dir, crop_size)
    evaluation_seed = _derive_seed(
        base_seed, 'shared_evaluation', 'transformations')

    evaluation_loader = create_dataloader(
        root_dir,
        crop_size,
        batch_size,
        num_workers,
        seed=evaluation_seed,
        dataset=full_dataset,
    )
    input_chunks = []
    accepted = 0
    for batch in evaluation_loader:
        valid = batch[
            batch[:, 0:1].mean(dim=(1, 2, 3)) > 0.15,
            0:1,
        ]
        if valid.numel():
            input_chunks.append(valid)
            accepted += len(valid)
        if accepted >= evaluation_samples:
            break
    if accepted < evaluation_samples:
        raise RuntimeError(
            f'Only {accepted} valid test presentations were available; '
            f'requested {evaluation_samples}.'
        )

    evaluation_inputs = torch.cat(input_chunks, dim=0)[:evaluation_samples]
    input_dimensionality, _ = get_pca_dimensions(
        evaluation_inputs, return_components=False)
    evaluation_hash = hashlib.sha256(
        evaluation_inputs.contiguous().numpy().tobytes()).hexdigest()
    metadata = {
        'sample_count': int(evaluation_samples),
        'post_training_presentations': True,
        'source_images_may_overlap_with_training': True,
        'transformations_are_separately_seeded': True,
        'training_file_count': len(full_dataset),
        'transformation_seed': int(evaluation_seed),
        'accepted_input_sha256': evaluation_hash,
        'brightness_threshold': 0.15,
        'input_pca_variance_threshold': 0.95,
        'input_pca_dimensionality': int(input_dimensionality),
    }
    result = (full_dataset, evaluation_inputs, metadata)
    _EVALUATION_CACHE[cache_key] = result
    return result


def _null_decoder_baseline(
        train_dataset,
        evaluation_inputs,
        crop_size,
        batch_size,
        num_workers,
        base_seed,
        epochs,
        device,
        n_decoders=3,
        constant_activity=0.05):
    """Train independent decoders whose cortical input is constant."""
    cache_key = (
        id(train_dataset), int(base_seed), int(epochs), int(n_decoders),
        float(constant_activity), str(device),
    )
    if cache_key in _NULL_BASELINE_CACHE:
        return _NULL_BASELINE_CACHE[cache_key]

    networks = []
    initialization_seeds = []
    for decoder_index in range(n_decoders):
        seed = _derive_seed(
            base_seed, 'null_decoder', 'initialization', decoder_index)
        initialization_seeds.append(int(seed))
        _seed_everything(seed)
        networks.append(
            init_nn(crop_size, crop_size, out_channels=1, device=device)
        )

    training_seed = _derive_seed(base_seed, 'null_decoder', 'training_loader')
    training_loader = create_dataloader(
        train_dataset.directory,
        crop_size,
        batch_size,
        num_workers,
        seed=training_seed,
        dataset=train_dataset,
    )
    for _ in range(epochs):
        for batch in tqdm(training_loader, leave=False, desc='null decoders'):
            targets = batch[
                batch[:, 0:1].mean(dim=(1, 2, 3)) > 0.15,
                0:1,
            ].to(device, non_blocking=True)
            if not len(targets):
                continue
            constant = torch.full_like(targets, constant_activity)
            for network in networks:
                reconstruction = network['activ'](
                    network['model'](constant))
                loss, _ = nn_loss(network, targets, reconstruction)
                network['optim'].zero_grad()
                loss.backward()
                network['optim'].step()

    scores = []
    with torch.no_grad():
        for network in networks:
            similarity_sum = 0.0
            for start in range(0, len(evaluation_inputs), 256):
                targets = evaluation_inputs[start:start + 256].to(
                    device, non_blocking=True)
                constant = torch.full_like(targets, constant_activity)
                reconstruction = network['activ'](
                    network['model'](constant))
                similarity = cosim(
                    targets, reconstruction, compute_on_input_device=True)
                similarity_sum += float(similarity) * len(targets)
            scores.append(similarity_sum / len(evaluation_inputs))

    result = {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'n_decoders': int(n_decoders),
        'constant_activity': float(constant_activity),
        'epochs': int(epochs),
        'training_seed': int(training_seed),
        'initialization_seeds': initialization_seeds,
        'input_size': int(crop_size),
    }
    _NULL_BASELINE_CACHE[cache_key] = result
    return result


def _evaluate_clean_model(
        model, network, evaluation_inputs, batch_size, device):
    """Return decoder accuracy and settled codes for one frozen model."""
    code_chunks = []
    accuracy_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(evaluation_inputs), batch_size):
            input_batch = evaluation_inputs[start:start + batch_size].to(
                device, non_blocking=True)
            response_batch = []
            for image in input_batch:
                _run_model(model, image[None], adaptation=False)
                response_batch.append(model.current_response.clone())
            responses = torch.cat(response_batch, dim=0)
            reconstruction = network['activ'](network['model'](responses))
            similarity = cosim(
                input_batch, reconstruction, compute_on_input_device=True)
            accuracy_sum += float(similarity) * len(input_batch)
            code_chunks.append(responses.cpu())
    return accuracy_sum / len(evaluation_inputs), torch.cat(code_chunks, dim=0)


# python3 -c 'import stats_collector; stats_collector.collect_stats()' -m wakepy
#@profile
def collect_stats(
        n_reps=3,
        trials=12,
        output_file=None,
        base_seed=DEFAULT_BASE_SEED,
        strict_determinism=STRICT_DETERMINISM):
    n_reps = _validate_n_reps(n_reps)
    trials = int(trials)
    if trials < 1:
        raise ValueError('trials must be at least 1')
    base_seed = int(base_seed)
    _configure_reproducibility(strict_determinism)
    _seed_everything(base_seed)
    
    # Example usage
    crop_size = 24 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = os.environ.get(
        'NEURALSHEET_INPUT_STIMULI', './input_stimuli')
    device = DEVICE  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 7
    beta = 1 - 5e-5
    loss_beta = 1e-2

    epochs = 2
    reproducibility = _reproducibility_context(
        root_dir, base_seed, strict_determinism)
    dataset, evaluation_inputs, evaluation_metadata = (
        _fixed_train_and_evaluation_data(
            root_dir, crop_size, batch_size, num_workers, base_seed)
    )
    null_baseline = _null_decoder_baseline(
        dataset,
        evaluation_inputs,
        crop_size,
        batch_size,
        num_workers,
        base_seed,
        epochs,
        device,
    )
    dataloader = create_dataloader(
        root_dir,
        crop_size,
        batch_size,
        num_workers,
        seed=_derive_seed(base_seed, 'collect_stats', 'shape_loader'),
        dataset=dataset,
    )
    
    trialvar = np.sqrt(np.linspace(2**2, 15**2, trials))
    sizesvar = [24,48,72]
    N_CODES = (sizesvar[-1]+1)**2
    sizes = len(sizesvar)
    if output_file is None:
        output_file = 'data_l4/data_accdim.pt'
    lr_initial = 1e-3
    lr_floor = 1e-4
    hebbian_lr_scale = 1e2
    model_kwargs = {
        'microcolumnar': False,
        'layer_3': False,
    }
    reco_tracker = torch.zeros((n_reps, sizes, trials, len(dataloader)))
    se_tracker = torch.zeros((n_reps, sizes, trials))
    spectrum_tracker = torch.zeros((n_reps, sizes, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((n_reps, sizes, trials))

    se_pca_tracker = torch.zeros((n_reps, sizes, trials))
    heldout_accuracy_tracker = torch.zeros((n_reps, sizes, trials))
    seed_records = []

    #------------------------- running simulations
    for s in range(sizes):
        
        for t in range(trials):

            for rep in range(n_reps):
                run_seed = _derive_seed(
                    base_seed, 'collect_stats', 'topological', s, t, rep)
                seeds = {
                    'run': run_seed,
                    'model': _derive_seed(run_seed, 'model'),
                    'decoder': _derive_seed(run_seed, 'decoder'),
                    'training': _derive_seed(run_seed, 'training'),
                    'train_dataloader': _derive_seed(
                        run_seed, 'train_dataloader'),
                }
                seed_records.append({
                    'rep_index': rep,
                    'size_index': s,
                    'radius_index': t,
                    'sheet_size': int(sizesvar[s]),
                    'interaction_radius': float(trialvar[t]),
                    'seeds': seeds,
                })

                dataloader = create_dataloader(
                    root_dir,
                    crop_size,
                    batch_size,
                    num_workers,
                    seed=seeds['train_dataloader'],
                    dataset=dataset,
                )
                print('cropsize: ', crop_size)

                print('running simulation rep: ' + str(rep + 1) + '/' + str(n_reps) \
                      + ' size: ' + str(float(sizesvar[s])) \
                      + ' interaction radius: ' + str(float(trialvar[t])) \
                      + ' run seed: ' + str(run_seed))

                _seed_everything(seeds['model'])
                model = NeuralSheet(crop_size, sizesvar[s], R_rf, R_long=trialvar[t], device=device, microcolumnar=False).to(device)
                    
                lr = lr_initial
                _seed_everything(seeds['decoder'])
                network = init_nn(sizesvar[s], crop_size, out_channels=1)
                _seed_everything(seeds['training'])
                avg_loss = 0
                batch_responses = []
                batch_inputs = []
                _cleanup()
            
                for e in range(epochs):
            
                    batch_progress = tqdm(dataloader, leave=False)
                    for b_idx, batch in enumerate(batch_progress):
            
                        del batch_inputs, batch_responses
                        batch_responses = []
                        batch_inputs = []
                        _cleanup()
                        
                        batch = batch.to(device, non_blocking=True)
                        # Preserve the per-image threshold and ordering while
                        # avoiding one CUDA synchronization for every crop.
                        valid_images = batch[
                            batch[:, 0:1].mean(dim=(1, 2, 3)) > 0.15
                        ]
            
                        for image in valid_images:
            
                            image = image[0:1][None].flip(1)
            
                            limit = lr_floor
                            lr *= beta
                            lr = lr if lr>limit else limit

                            model.hebbian_lr = lr * hebbian_lr_scale
                            model.homeo_lr = lr

                            _run_model(model, image)
                            model.hebbian_step()

                            response = model.current_response.clone()
                            batch_responses.append(response)
                            batch_inputs.append(model.current_input.clone())
            
                        if len(batch_responses):
                            
                            batch_responses = torch.cat(batch_responses, dim=0)
                            batch_inputs = torch.cat(batch_inputs, dim=0)

                            # The model learns from every valid crop, as before;
                            # only zero codes are omitted from decoder/PCA data.
                            nonzero_responses = (
                                batch_responses.sum(dim=(1, 2, 3)) != 0
                            )
                            if not bool(nonzero_responses.any()):
                                continue

                            batch_responses = batch_responses[nonzero_responses]
                            batch_inputs = batch_inputs[nonzero_responses]
                            reco_input = network['activ'](network['model'](batch_responses))
            
                            targets = batch_inputs
                            loss, loss_std = nn_loss(network, targets, reco_input)
            
                            sim = cosim(
                                targets.detach(),
                                reco_input.detach(),
                                compute_on_input_device=True,
                            )
                            reco_tracker[rep, s, t, b_idx] = sim
            
                            avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim
            
                            network['optim'].zero_grad()
                            loss.backward()
                            network['optim'].step()
            
                            mean_activation = model.mean_activations.mean()
                            mean_std = model.mean_activations.std() / model.homeo_target
                            batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}'.format(
                                mean_activation, 
                                mean_std, 
                                avg_loss,
                                lr
                            ))

                #------------------------- held-out accuracy/dimensionality

                heldout_accuracy, heldout_codes = _evaluate_clean_model(
                    model,
                    network,
                    evaluation_inputs,
                    batch_size,
                    device,
                )
                mask = torch.isnan(heldout_codes).any(dim=(-2, -1))
                print('finished training, number of Nan found: ' + str(int(mask.sum())))

                eff_dims, spectrum, peak = get_effective_dims(
                    heldout_codes[-N_CODES:])
                eff_dims_pca, _ = get_pca_dimensions(
                    heldout_codes,
                    return_components=False,
                )

                se_tracker[rep, s, t] = eff_dims
                se_pca_tracker[rep, s, t] = eff_dims_pca
                heldout_accuracy_tracker[rep, s, t] = heldout_accuracy

                print('training complete, held-out accuracy: ' + str(heldout_accuracy) + ' dimensionality: ' \
                      + str(eff_dims_pca))

                spectrum_tracker[rep,s,t,:sizesvar[s],:sizesvar[s]] = spectrum.cpu()
                peak_tracker[rep,s,t] = peak.cpu()                    

                del heldout_codes, model, network, spectrum, peak
                del batch, batch_responses, batch_inputs, reco_input, targets
                _cleanup()

    reproducibility['seed_records'] = seed_records
    reproducibility['completed_at_utc'] = (
        datetime.now(timezone.utc).isoformat())

    config = {
        'experiment': 'accuracy_dimensionality',
        'output_file': output_file,
        'device': device,
        'root_dir': root_dir,
        'crop_size': crop_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'aggressive_cleanup': AGGRESSIVE_CLEANUP,
        'epochs': epochs,
        'n_reps': n_reps,
        'trials': trials,
        'sizesvar': _as_list(sizesvar),
        'trialvar': _as_list(trialvar),
        'N_CODES': int(N_CODES),
        'evaluation_samples': len(evaluation_inputs),
        'R_rf': R_rf,
        'beta': beta,
        'loss_beta': loss_beta,
        'lr_initial': lr_initial,
        'lr_floor': lr_floor,
        'hebbian_lr_scale': hebbian_lr_scale,
        'model_kwargs': model_kwargs,
        'evaluation': evaluation_metadata,
        'null_decoder': null_baseline,
        'collector_optimizations': {
            'reuse_dataset_index': True,
            'batched_validity_checks': True,
            'shared_response_copy': True,
            'cosine_compute_device': device,
            'discard_unused_pca_components': True,
            'l4_only_interaction_channels': True,
        },
        'reproducibility': reproducibility,
        'decoder': {
            'init_fn': 'init_nn',
            'input_size': 'sheet_size',
            'output_size': crop_size,
            'out_channels': 1,
            'comparison_region': 'full_crop',
        },
        'result_axes': {
            'reco_tracker': ['rep', 'size', 'radius', 'batch'],
            'se_tracker': ['rep', 'size', 'radius'],
            'spectrum_tracker': ['rep', 'size', 'radius', 'x', 'y'],
            'peak_tracker': ['rep', 'size', 'radius'],
            'se_pca_tracker': ['rep', 'size', 'radius'],
            'heldout_accuracy': ['rep', 'size', 'radius'],
        },
    }

    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'se_pca_tracker': se_pca_tracker,
        'heldout_accuracy': heldout_accuracy_tracker,
        'input_pca_dimensionality': int(
            evaluation_metadata['input_pca_dimensionality']),
        'null_accuracy_scores': torch.tensor(null_baseline['scores']),
        'null_accuracy_mean': float(null_baseline['mean']),
        'trialvar': trialvar,
        'sizesvar': sizesvar,
        'n_reps': n_reps,
        'config': config
    }


    torch.save(data, output_file)
    time.sleep(5)

def collect_noise_stats(
        minicolumnar=True,
        n_reps=3,
        trials=12,
        output_file=None,
        base_seed=DEFAULT_BASE_SEED,
        strict_determinism=STRICT_DETERMINISM):
    n_reps = _validate_n_reps(n_reps)
    trials = int(trials)
    if trials < 1:
        raise ValueError('trials must be at least 1')
    base_seed = int(base_seed)
    _configure_reproducibility(strict_determinism)
    _seed_everything(base_seed)
    
    # Example usage
    crop_size = 24 # Crop size (NxN)
    batch_size = 32  # Number of crops to load at once
    num_workers = 4  # Number of threads for data loading
    root_dir = os.environ.get(
        'NEURALSHEET_INPUT_STIMULI', './input_stimuli')
    device = DEVICE  # Assuming CUDA is available and desired
    #M = 56  # Neural sheet dimensions
    #std_exc = 0.25 # Standard deviation for excitation Gaussian
    R_rf = 7
    beta = 1 - 5e-5
    loss_beta = 1e-2

    architecture = (
        'salt_and_pepper' if minicolumnar else 'topological')
    reproducibility = _reproducibility_context(
        root_dir, base_seed, strict_determinism)
    epochs = 2
    dataset, evaluation_inputs, evaluation_metadata = (
        _fixed_train_and_evaluation_data(
            root_dir, crop_size, batch_size, num_workers, base_seed)
    )
    null_baseline = _null_decoder_baseline(
        dataset,
        evaluation_inputs,
        crop_size,
        batch_size,
        num_workers,
        base_seed,
        epochs,
        device,
    )
    dataloader = create_dataloader(
        root_dir,
        crop_size,
        batch_size,
        num_workers,
        seed=_derive_seed(
            base_seed, 'collect_noise_stats', architecture, 'shape_loader'),
        dataset=dataset,
    )
    
    n_conditions = 15

    #if minicolumnar:
    trialvar = np.sqrt(np.linspace(2**2, 15**2, trials))
    #else:
    #    trialvar = np.sqrt(np.linspace(2**2, 18**2, trials))
    
    sizesvar = np.round(trialvar * 5).astype(int)
    noise_conditions = np.linspace(0, 0.1, n_conditions)
    N_CODES = int((sizesvar[-1]+1)**2)
    robustness_samples = len(evaluation_inputs)
    if output_file is None:
        output_file = (
            'data_l4/data_noise_sp.pt'
            if minicolumnar else 'data_l4/data_noise_topo.pt'
        )
    lr_initial = 1e-3
    lr_floor = 1e-4
    hebbian_lr_scale = 1e2
    model_kwargs = {
        'microcolumnar': bool(minicolumnar),
        'layer_3': False,
    }
    reco_tracker = torch.zeros((n_reps, trials, len(dataloader)))
    se_tracker = torch.zeros((n_reps, trials))
    spectrum_tracker = torch.zeros((n_reps, trials, sizesvar[-1], sizesvar[-1]))
    peak_tracker = torch.zeros((n_reps, trials))

    se_pca_tracker = torch.zeros((n_reps, trials))

    # Accuracy and dimensionality are measured only for the clean condition.
    # NaN explicitly marks the intentionally unmeasured noisy conditions.
    noise_acc_tracker = torch.full((n_reps, trials, n_conditions), torch.nan)
    noise_dim_tracker = torch.full((n_reps, trials, n_conditions), torch.nan)
    noise_rob_tracker = torch.full(
        (n_reps, trials, n_conditions), torch.nan)
    seed_records = []

    #------------------------- running simulations
    for t in range(trials):

        for rep in range(n_reps):
            run_seed = _derive_seed(
                base_seed, 'collect_noise_stats', architecture, t, rep)
            seeds = {
                'run': run_seed,
                'model': _derive_seed(run_seed, 'model'),
                'decoder': _derive_seed(run_seed, 'decoder'),
                'training': _derive_seed(run_seed, 'training'),
                'train_dataloader': _derive_seed(
                    run_seed, 'train_dataloader'),
                'noise_conditions': [
                    _derive_seed(run_seed, 'noise_condition', n_idx)
                    for n_idx in range(n_conditions)
                ],
            }
            seed_record = {
                'rep_index': rep,
                'radius_index': t,
                'sheet_size': int(sizesvar[t]),
                'interaction_radius': float(trialvar[t]),
                'architecture': architecture,
                'seeds': seeds,
            }
            seed_records.append(seed_record)

            dataloader = create_dataloader(
                root_dir,
                crop_size,
                batch_size,
                num_workers,
                seed=seeds['train_dataloader'],
                dataset=dataset,
            )
            print('cropsize: ', crop_size)

            print('running simulation rep: ' + str(rep + 1) + '/' + str(n_reps) \
                  + ' size: ' + str(float(sizesvar[t])) \
                  + ' interaction radius: ' + str(float(trialvar[t])) \
                  + ' run seed: ' + str(run_seed))

            _seed_everything(seeds['model'])
            if minicolumnar:
                model = NeuralSheet(crop_size, int(sizesvar[t]), R_rf, R_long=trialvar[t], device=device, microcolumnar=True).to(device)
            else:
                model = NeuralSheet(crop_size, int(sizesvar[t]), R_rf, R_long=trialvar[t], device=device, microcolumnar=False).to(device)
                
            lr = lr_initial
            _seed_everything(seeds['decoder'])
            network = init_nn(int(sizesvar[t]), crop_size, out_channels=1)
            _seed_everything(seeds['training'])
            avg_loss = 0
            batch_responses = []
            batch_inputs = []
            _cleanup()
        
            for e in range(epochs):

                batch_progress = tqdm(dataloader, leave=False)
                for b_idx, batch in enumerate(batch_progress):


                    batch_responses = []
                    batch_inputs = []
                    batch = batch.to(device)  # Transfer the entire batch to GPU

                    for image in batch:

                        image = image[0:1][None].flip(1)

                        if image.mean()>0.15:

                            limit = lr_floor
                            lr *= beta
                            lr = lr if lr>limit else limit
                            model.hebbian_lr = lr * hebbian_lr_scale
                            model.homeo_lr = lr

                            _run_model(model, image, adaptation=True)
                            model.hebbian_step()
                            
                            batch_responses.append(model.current_response.clone())
                            batch_inputs.append(model.current_input.clone())
                    batch_responses = _cat_or_none(batch_responses)
                    batch_inputs = _cat_or_none(batch_inputs)

                    if batch_responses is None:
                        continue

                    reco_input = network['activ'](network['model'](batch_responses))
                    targets = batch_inputs
                    
                    loss, loss_std = nn_loss(network, targets, reco_input)
                    
                    # Match collect_stats() and the post-training noise
                    # measurements: average each sample's cosine equally.
                    sim = cosim(targets.detach(), reco_input.detach())
                    reco_tracker[rep, t, b_idx] = sim
                                    
                    avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim
                    
                    network['optim'].zero_grad()
                    loss.backward()
                    network['optim'].step()

                    mean_activation = model.mean_activations.mean()
                    mean_std = model.mean_activations.std() / model.homeo_target
                    
                    batch_progress.set_description('M:{:.3f} STD:{:.3f} BCE:{:.3f} LR:{:.5f} SP:{:.3f} D:{:.3f} A:{:.3f}'.format(
                        mean_activation, 
                        mean_std, 
                        avg_loss,
                        lr,
                        model.aff_gain.mean(),
                        model.lat_gain.mean(),
                        model.delta_mag.mean()
                    ))
                    
            # One frozen, file-disjoint set supplies all three measurements.
            clean_accuracy, clean_codes = _evaluate_clean_model(
                model,
                network,
                evaluation_inputs,
                batch_size,
                device,
            )
            mask = torch.isnan(clean_codes).any(dim=(-2, -1))
            print('finished training, number of NaNs found: ' + str(int(mask.sum())))

            eff_dims, spectrum, peak = get_effective_dims(
                clean_codes[-N_CODES:])
            clean_eff_dims_pca, _ = get_pca_dimensions(
                clean_codes, return_components=False)
            se_tracker[rep, t] = eff_dims
            se_pca_tracker[rep, t] = clean_eff_dims_pca
            noise_dim_tracker[rep, t, 0] = clean_eff_dims_pca
            noise_acc_tracker[rep, t, 0] = clean_accuracy
            noise_rob_tracker[rep, t, 0] = 1.0
            sheet_size = int(sizesvar[t])
            spectrum_tracker[rep,t,:sheet_size,:sheet_size] = spectrum.cpu()
            peak_tracker[rep,t] = peak.cpu()
            print(
                'held-out reconstruction accuracy: '
                + str(clean_accuracy)
                + ', dimensionality: '
                + str(clean_eff_dims_pca)
            )

            print('collecting noise robustness measurements!')
            p = int(trialvar[t] // 2)

            for n_idx, noise_gamma in tqdm(
                    enumerate(noise_conditions[1:], start=1),
                    total=n_conditions - 1):
                _seed_everything(seeds['noise_conditions'][n_idx])
                robustness_sum = torch.zeros((), device=device)

                with torch.no_grad():
                    for batch_start in range(
                            0, robustness_samples, batch_size):
                        batch_end = min(
                            batch_start + batch_size, robustness_samples)
                        input_batch = evaluation_inputs[
                            batch_start:batch_end].to(device)
                        clean_batch = clean_codes[
                            batch_start:batch_end].to(device)

                        for image_idx, image in enumerate(input_batch):
                            _run_model(
                                model,
                                image[None],
                                noise_gamma=noise_gamma,
                                adaptation=False
                            )

                            clean_response = clean_batch[
                                image_idx:image_idx+1,
                                :,
                                p:-p-1,
                                p:-p-1
                            ].flatten(1)
                            perturbed_response = model.current_response[
                                :,
                                :,
                                p:-p-1,
                                p:-p-1
                            ].flatten(1)
                            numerator = (
                                clean_response * perturbed_response).sum(1)
                            denominator = torch.sqrt(
                                (clean_response**2).sum(1)
                                * (perturbed_response**2).sum(1)
                            )
                            robustness_sum += (
                                numerator / (denominator + 1e-11)).sum()

                robustness = float(robustness_sum.cpu()) / robustness_samples
                noise_rob_tracker[rep, t, n_idx] = robustness

                print(
                    'measuring noise robustness, noise: '
                    + str(noise_gamma)
                    + ', robustness: '
                    + str(robustness)
                )

                if robustness < 0.9:
                    break

    reproducibility['seed_records'] = seed_records
    reproducibility['completed_at_utc'] = (
        datetime.now(timezone.utc).isoformat())

    config = {
        'experiment': 'noise_robustness',
        'output_file': output_file,
        'device': device,
        'root_dir': root_dir,
        'crop_size': crop_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'aggressive_cleanup': AGGRESSIVE_CLEANUP,
        'epochs': epochs,
        'n_reps': n_reps,
        'trials': trials,
        'n_conditions': n_conditions,
        'sizesvar': _as_list(sizesvar),
        'trialvar': _as_list(trialvar),
        'noise_conditions': _as_list(noise_conditions),
        'N_CODES': int(N_CODES),
        'robustness_samples': robustness_samples,
        'R_rf': R_rf,
        'beta': beta,
        'loss_beta': loss_beta,
        'lr_initial': lr_initial,
        'lr_floor': lr_floor,
        'hebbian_lr_scale': hebbian_lr_scale,
        'minicolumnar': bool(minicolumnar),
        'model_kwargs': model_kwargs,
        'evaluation': evaluation_metadata,
        'null_decoder': null_baseline,
        'reproducibility': reproducibility,
        'decoder': {
            'init_fn': 'init_nn',
            'input_size': 'sheet_size',
            'output_size': crop_size,
            'out_channels': 1,
            'comparison_region': 'full_crop',
        },
        'robustness': {
            'activity_margin': 'int(trialvar[t] // 2)',
            'early_stop_robustness': 0.9,
            'fixed_clean_sample_set': True,
            'same_sample_for_fidelity_complexity_robustness': True,
            'source_images_may_overlap_with_training': True,
            'test_transformations_are_separately_seeded': True,
            'reuse_clean_responses': True,
            'independent_noise_per_condition': True,
            'se_pca_source': 'fixed_clean_robustness_sample',
            'accuracy_dimensionality_noise_indices': [0],
            'unmeasured_accuracy_dimensionality_value': 'NaN',
        },
        'result_axes': {
            'reco_tracker': ['rep', 'radius', 'batch'],
            'se_tracker': ['rep', 'radius'],
            'spectrum_tracker': ['rep', 'radius', 'x', 'y'],
            'peak_tracker': ['rep', 'radius'],
            'se_pca_tracker': ['rep', 'radius'],
            'noise_acc': ['rep', 'radius', 'noise_condition'],
            'noise_dim': ['rep', 'radius', 'noise_condition'],
            'noise_rob': ['rep', 'radius', 'noise_condition'],
        },
    }

    data = {
        'reco_tracker' : reco_tracker,
        'se_tracker' : se_tracker,
        'spectrum_tracker': spectrum_tracker,
        'peak_tracker': peak_tracker,
        'se_pca_tracker': se_pca_tracker,
        'trialvar': trialvar,
        'sizesvar': sizesvar,
        'n_reps': n_reps,
        'noise_conditions' : noise_conditions,
        'noise_acc': noise_acc_tracker,
        'noise_dim': noise_dim_tracker,
        'noise_rob': noise_rob_tracker,
        'input_pca_dimensionality': int(
            evaluation_metadata['input_pca_dimensionality']),
        'null_accuracy_scores': torch.tensor(null_baseline['scores']),
        'null_accuracy_mean': float(null_baseline['mean']),
        'config': config
    }


    if minicolumnar:
        torch.save(data, output_file)
    else:
        torch.save(data, output_file)
        
    time.sleep(5)

            
def train_map(sheet_size, crop_size, epochs, dataloader, beta, model, reco_tracker=None, loss_beta=1e-2):
    lr = 1e-3
    network = init_nn(sheet_size, crop_size, out_channels=1)
    avg_loss = 0
    code_tracker = []
    batch_responses = []
    batch_inputs = []
    _cleanup()

    for e in range(epochs):

        batch_progress = tqdm(dataloader, leave=False)
        del code_tracker
        code_tracker = []
        
        for b_idx, batch in enumerate(batch_progress):

            del batch_inputs, batch_responses
            batch_responses = []
            batch_inputs = []
            _cleanup()
            
            batch = batch.to(DEVICE)  # Transfer the entire batch to GPU

            for image in batch:

                image = image[0:1][None].flip(1)

                if image.mean()>0.15:

                    limit = 1e-4
                    lr *= beta
                    lr = lr if lr>limit else limit

                    model.hebbian_lr = lr
                    model.homeo_lr = lr

                    _run_model(model, image)
                    model.hebbian_step()

                    if model.current_response.sum():

                        batch_responses.append(model.current_response.clone())
                        batch_inputs.append(model.current_input.clone())
                        code_tracker.append(model.current_response.clone())

            if len(batch_responses):
                
                batch_responses = torch.cat(batch_responses, dim=0)
                batch_inputs = torch.cat(batch_inputs, dim=0)

                reco_input = network['activ'](network['model'](batch_responses))

                targets = batch_inputs
                loss, loss_std = nn_loss(network, targets, reco_input)

                sim = cosim(targets.detach(), reco_input.detach())
                if reco_tracker is not None:
                    reco_tracker[b_idx] = sim

                avg_loss = (1-loss_beta)*avg_loss + loss_beta*sim

                network['optim'].zero_grad()
                loss.backward()
                network['optim'].step()

                if b_idx%50==0:
                    ori_map, phase_map, mean_tc = get_orientations(
                        model.afferent_weights, gabor_size=model.rf_size)

                mean_activation = model.mean_activations.mean()
                mean_std = model.mean_activations.std() / model.homeo_target
                batch_progress.set_description('M:{:.3f}, STD:{:.3f}, BCE:{:.3f}, LR:{:.5f}, AS:{:.3f}'.format(
                    mean_activation, 
                    mean_std, 
                    avg_loss,
                    lr,
                    model.aff_gain.mean()
                ))

    return model, network, code_tracker


def run_noise_sweeps(
        n_reps=3,
        trials=12,
        output_files=None,
        base_seed=DEFAULT_BASE_SEED,
        strict_determinism=STRICT_DETERMINISM):
    if output_files is None:
        output_files = (None, None)
    with keep.running():
        collect_noise_stats(
            minicolumnar=True,
            n_reps=n_reps,
            trials=trials,
            output_file=output_files[0],
            base_seed=base_seed,
            strict_determinism=strict_determinism,
        )
        collect_noise_stats(
            minicolumnar=False,
            n_reps=n_reps,
            trials=trials,
            output_file=output_files[1],
            base_seed=base_seed,
            strict_determinism=strict_determinism,
        )

if __name__ == '__main__':
    run_noise_sweeps()
