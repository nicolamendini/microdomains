import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.wiring_efficiency_utils import *

class NeuralSheet(nn.Module):
    def __init__(
        self,
        input_size,
        sheet_size,
        R_rf,
        R_long,
        homeo_target=0.05,
        act_target=0.26,
        aff_baseline=0.3,
        lat_dom=0.5,
        loc_b=0.4,
        lat_dom_l3=0.,
        iterations=30,
        lr=1e-3,
        hebbian_lr_ratio=100,
        microcolumnar=False,
        device='cuda',
    ):
        super().__init__()

        self.act_target = act_target
        self.aff_baseline = aff_baseline
        self.lat_dom = lat_dom

        self.loc_b = loc_b
        lat_dom_l3 = lat_dom_l3 if lat_dom_l3 is not None else self.lat_dom
        self.lat_dom_l3 = lat_dom_l3

        self.homeo_lr = lr
        self.hebbian_lr_ratio = hebbian_lr_ratio
        self.hebbian_lr = lr * hebbian_lr_ratio
        self.homeo_target = homeo_target
        self.iterations = iterations
        self.sheet_size = sheet_size
        self.input_size = input_size
        self.device = device
        self.microcolumnar = microcolumnar
        self._last_forward_layer_3 = False

        self.rf_size = oddenise(R_rf*2)
        self.rf_size_l3 = oddenise(R_long*2/1.8)
        self.aff_pad = self.rf_size
        self.R_long = R_long

        self.aff_cutoff = get_circle(self.rf_size, self.rf_size/2).float().to(device)
        self.aff_cutoff_l3 = get_circle(self.rf_size_l3, self.rf_size_l3/2).float().to(device)

        a=1.8
        b=1.5
        c=3

        if microcolumnar:
            self.se_cutoff = generate_circles(sheet_size, sheet_size, R_long/a/b/c).to(device)
            self.i_cutoff = generate_circles(sheet_size, sheet_size, R_long/a/b).to(device)
            self.le_cutoff = generate_circles(sheet_size, sheet_size, R_long/a).to(device)
            #envelope = generate_gaussians(sheet_size, sheet_size, R_long/a/1.5).to(device)
            #self.le_cutoff *= torch.rand(self.le_cutoff.shape, device=device) + 0
            #envelope /= envelope.max()
            #self.le_cutoff *= envelope
            self.exc_b = 0.25

        else:
            self.se_cutoff = generate_circles(sheet_size, sheet_size, R_long/c).to(device)
            self.i_cutoff = generate_circles(sheet_size, sheet_size, R_long).to(device)

        self.s_exc = self.se_cutoff / self.se_cutoff.sum([2,3], keepdim=True)

        afferent_weights = torch.rand((sheet_size**2, 2, self.rf_size, self.rf_size), device=device)
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights

        afferent_weights_l3 = torch.rand((sheet_size**2, 1, self.rf_size_l3, self.rf_size_l3), device=device)
        afferent_weights_l3 /= afferent_weights_l3.sum([2,3], keepdim=True)
        self.afferent_weights_l3 = afferent_weights_l3

        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations + 0
        self.lateral_correlations_exc = lateral_correlations + 0
        self.lateral_correlations_l3 = lateral_correlations + 0
        self.lateral_correlations_exc_l3 = lateral_correlations + 0
        self.lateral_correlations_l4_l3 = lateral_correlations + 0
        self.lateral_correlations_exc_l4_l3 = lateral_correlations + 0

        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)
        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_lat = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.lat_gain = torch.ones(1, 1, 1, 1, device=device)
        self.mean_aff = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_gain = torch.ones(1, 1, 1, 1, device=device)
        self.mix = torch.ones(1, 1, 1, 1, device=device) * self.lat_dom
        self.avg_hist = torch.zeros(10, device=device)
        self.old_style_mean_fr = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.old_style_mean_aff = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.frozen_l4_inh_sparsity = torch.zeros_like(self.lateral_correlations, dtype=torch.bool)
        self.frozen_l4_sparsity = torch.tensor(-1.0, device=device)
        self.l4_sparsity_freeze_active = torch.tensor(False, device=device)
        self.l4_inh_sparsity = 1
        self.noise = 0

        self.current_response_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.response_tracker_l3 = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)
        self.mean_activations_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device) + self.homeo_target
        self.thresholds_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_fr_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_lat_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.lat_gain_l3 = torch.ones(1, 1, 1, 1, device=device)
        self.mean_aff_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.aff_gain_l3 = torch.ones(1, 1, 1, 1, device=device)
        self.mix_l3 = torch.ones(1, 1, 1, 1, device=device) * self.lat_dom_l3
        self.avg_hist_l3 = torch.zeros(10, device=device)
        self.old_style_mean_fr_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.old_style_mean_aff_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.global_exc_threshold_l3 = torch.ones(1, 1, sheet_size, sheet_size, device=device) * 0.06
        self.global_inh_threshold_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.rw_global_exc_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.rw_global_inh_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.rw_global_net_l3 = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.target_rw_global_exc_l3 = 0.2
        self.target_rw_global_inh_l3 = 0.15
        self.noise_l3 = 0
        self.frozen_global_exc_sparsity_l3 = torch.zeros_like(self.lateral_correlations_exc_l3, dtype=torch.bool)
        self.frozen_global_inh_sparsity_l3 = torch.zeros_like(self.lateral_correlations_l3, dtype=torch.bool)
        self.frozen_global_sparsity_l3 = torch.tensor(-1.0, device=device)
        self.global_sparsity_freeze_active_l3 = torch.tensor(False, device=device)

        r = int(self.R_long//a+1 if self.microcolumnar else self.R_long)
        self.window = oddenise(r*2)

        self.init_indices()
        self.rf_grids = get_grids(input_size, input_size, self.rf_size, sheet_size, jitter=0, device=device)
        self.rf_grids_l3 = get_grids(sheet_size, sheet_size, self.rf_size_l3, sheet_size, device=device)

        self.slicing_var = torch.zeros(
            (self.sheet_size**2, 5, sheet_size+r*2, sheet_size+r*2),
            device=device
        )

        self.delta_mag = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.eye = torch.eye(self.sheet_size**2).view(self.lateral_correlations.shape).to(device)

    def forward(
        self,
        input_crop,
        noise_gamma=0,
        noise_beta=0.8,
        adaptation=True,
        sparsity=0,
        sparsity_freeze=False,
        l4_sparsity=0,
        l4_sparsity_freeze=False,
        loc_sparsity=0,
        layer_3=True,
        track_response=False,
    ):

        self._last_forward_layer_3 = bool(layer_3)
        self.current_response *= 0
        if layer_3:
            self.current_response_l3 *= 0
        if track_response:
            self.response_tracker *= 0
            if layer_3:
                self.response_tracker_l3 *= 0
        if layer_3:
            self.current_tiles_l3 = 0

        self.current_input = input_crop
        self.current_tiles = extract_patches(input_crop, self.rf_grids)
        afferent = self.current_tiles * self.get_aff_weights()
        afferent = afferent.sum([1,2,3])
        self.current_afferent = afferent.view(self.current_response.shape)

        self._ensure_l4_sparsity_freeze_state()
        if l4_sparsity:
            if l4_sparsity_freeze:
                sparsity_value = float(l4_sparsity)
                needs_frozen_mask = (
                    not bool(self.l4_sparsity_freeze_active.item())
                    or abs(float(self.frozen_l4_sparsity.item()) - sparsity_value) > 1e-6
                )
                if needs_frozen_mask:
                    self.frozen_l4_inh_sparsity = self._weighted_connection_mask(
                        self.lateral_correlations,
                        l4_sparsity,
                        eligible_mask=self.i_cutoff > 0,
                    )
                    self.frozen_l4_sparsity.fill_(sparsity_value)
                    self.l4_sparsity_freeze_active.fill_(True)
                self.l4_inh_sparsity = self.frozen_l4_inh_sparsity
            else:
                self.l4_inh_sparsity = self._weighted_connection_mask(
                    self.lateral_correlations,
                    l4_sparsity,
                    eligible_mask=self.i_cutoff > 0,
                )
        else:
            self.l4_inh_sparsity = 1

        self.lat_sparsity = 1
        if layer_3:
            self._ensure_global_sparsity_freeze_state()
            if sparsity:
                if sparsity_freeze:
                    sparsity_value = float(sparsity)
                    needs_frozen_mask = (
                        not bool(self.global_sparsity_freeze_active_l3.item())
                        or abs(float(self.frozen_global_sparsity_l3.item()) - sparsity_value) > 1e-6
                    )
                    if needs_frozen_mask:
                        self.frozen_global_exc_sparsity_l3 = self._weighted_connection_mask(
                            self.lateral_correlations_exc_l3,
                            sparsity,
                        )
                        self.frozen_global_inh_sparsity_l3 = self._weighted_connection_mask(
                            self.lateral_correlations_l3,
                            sparsity,
                        )
                        self.frozen_global_sparsity_l3.fill_(sparsity_value)
                        self.global_sparsity_freeze_active_l3.fill_(True)
                    self.global_exc_sparsity_l3 = self.frozen_global_exc_sparsity_l3
                    self.global_inh_sparsity_l3 = self.frozen_global_inh_sparsity_l3
                else:
                    self.global_exc_sparsity_l3 = self._weighted_connection_mask(
                        self.lateral_correlations_exc_l3,
                        sparsity,
                    )
                    self.global_inh_sparsity_l3 = self._weighted_connection_mask(
                        self.lateral_correlations_l3,
                        sparsity,
                    )
            else:
                self.global_exc_sparsity_l3 = 1
                self.global_inh_sparsity_l3 = 1

            if loc_sparsity:
                self.loc_lat_sparsity = torch.rand(self.lateral_correlations.shape, device=self.device) < (1-loc_sparsity)
            else:
                self.loc_lat_sparsity = 1

        self.update_interactions(layer_3=layer_3)

        pad_amount = self.window // 2

        se_padded = F.pad(self.s_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
        self.slicing_var[:, 0:1] = se_padded

        i_padded = F.pad(self.inh, (pad_amount, pad_amount, pad_amount, pad_amount))
        self.slicing_var[:, 1:2] = i_padded

        if self.microcolumnar:

            le_padded = F.pad(self.l_exc, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 2:3] = le_padded / le_padded.sum([1,2,3], keepdim=True)

        if layer_3:

            i_padded_l4_l3 = F.pad(self.inh_l4_l3, (pad_amount, pad_amount, pad_amount, pad_amount))
            self.slicing_var[:, 3:4] = i_padded_l4_l3

            if self.microcolumnar:

                le_padded_l4_l3 = F.pad(self.l_exc_l4_l3, (pad_amount, pad_amount, pad_amount, pad_amount))
                self.slicing_var[:, 4:5] = le_padded_l4_l3

        if layer_3:
            interaction_channels = 5
        elif self.microcolumnar:
            interaction_channels = 3
        else:
            interaction_channels = 2

        sliced_interactions = self.slicing_var[
            self.batch_indices,
            :interaction_channels,
            self.final_row_indices,
            self.final_col_indices,
        ]

        se_crops = sliced_interactions[:,:,:,:,0]
        i_crops = sliced_interactions[:,:,:,:,1]

        if self.microcolumnar:
            le_crops = sliced_interactions[:,:,:,:,2]

            lat_interactions = 0.25*se_crops + 0.7*le_crops - i_crops

            if layer_3:
                i_crops_l4_l3 = sliced_interactions[:,:,:,:,3]
                le_crops_l4_l3 = sliced_interactions[:,:,:,:,4]
                local_int = self.s_exc_l3 - self.inh_l3
                l4_l3_int = 1.5*(0.25*se_crops + (1-0.7)*le_crops) - i_crops_l4_l3

        else:

            lat_interactions = se_crops - i_crops

            if layer_3:
                i_crops_l4_l3 = sliced_interactions[:,:,:,:,3]
                local_int = self.s_exc_l3 - self.inh_l3
                l4_l3_int = 1.5*se_crops - i_crops_l4_l3


        for i in range(self.iterations):

            if noise_gamma:

                if i==0:
                    self.noise = torch.randn(self.current_response.shape, device=self.device)
                    self.noise_l3 = torch.randn(self.current_response.shape, device=self.device)

                else:
                    curr_noise = torch.randn(self.current_response.shape, device=self.device)
                    self.noise = self.noise * noise_beta + curr_noise * (1-noise_beta)
                    self.noise /= (self.noise**2).mean()**0.5

                    curr_noise_l3 = torch.randn(self.current_response.shape, device=self.device)
                    self.noise_l3 = self.noise_l3 * noise_beta + curr_noise_l3 * (1-noise_beta)
                    self.noise_l3 /= (self.noise_l3**2).mean()**0.5

            else:
                self.noise = 0
                self.noise_l3 = 0

            padded_response = F.pad(self.current_response, (self.window//2, self.window//2, self.window//2, self.window//2))
            res_tiles = F.unfold(padded_response, self.window)[0].T.view(-1,1,self.window,self.window)
            mex_hat = (lat_interactions * res_tiles).sum([2,3]).view(self.current_response.shape)

            afferent_delta = (self.current_afferent - self.aff_baseline) * self.aff_gain
            lateral_delta = mex_hat * self.lat_gain

            if layer_3:

                afferent_drive_l3 = (l4_l3_int * res_tiles).sum([2,3]).view(self.current_response.shape)
                afferent_delta_l3 = afferent_drive_l3 * self.aff_gain_l3

                global_int = self.global_exc_l3 - self.global_inh_l3
                interaction_l3 = global_int * (1-self.loc_b) + local_int * self.loc_b
                lateral_delta_l3 = (interaction_l3 * self.current_response_l3).sum([1,2,3]).view(self.current_response_l3.shape)

            update = lateral_delta + afferent_delta - self.thresholds
            self.current_response = torch.tanh(torch.relu(update + self.noise * noise_gamma))
            
            if track_response:
                self.response_tracker[i] = self.current_response

            if layer_3:

                delta_l3 = lateral_delta_l3 * self.lat_gain_l3

                update_l3 = afferent_delta_l3 + delta_l3 #- self.thresholds_l3

                self.current_response_l3 = torch.tanh(torch.relu(update_l3 + self.noise_l3 * noise_gamma))

                if track_response:
                    self.response_tracker_l3[i] = self.current_response_l3


        if adaptation:
            if layer_3:
                self.current_tiles_l3 = extract_patches(self.current_response, self.rf_grids_l3)

            gated_response = self.current_response.detach() * self.homeo_lr * 10
            self.mean_lat = self.mean_lat * (1 - gated_response) + torch.relu(lateral_delta) * gated_response
            self.mean_aff = self.mean_aff * (1 - gated_response) + torch.relu(afferent_delta) * gated_response
            self.mean_fr = self.mean_lat + self.mean_aff
            self.mix = self.mean_lat.mean() / (self.mean_lat.mean() + self.mean_aff.mean() + 1e-11)
            # Branch-strength adaptation: aff_gain controls afferent energy,
            # lat_gain controls lateral energy; lat_dom only defines the split.

            target_aff = self.act_target * (1 - self.lat_dom)
            target_lat = self.act_target * self.lat_dom
            beta_error = (self.mean_aff.mean() - target_aff) / (target_aff + 1e-11)
            gain_error = (self.mean_lat.mean() - target_lat) / (target_lat + 1e-11)
            self.aff_gain -= beta_error * self.homeo_lr
            self.aff_gain = self.aff_gain.clip(0.1)
            self.lat_gain -= gain_error * self.homeo_lr
            self.lat_gain = self.lat_gain.clip(0.1)

            new_hist = self._activity_histogram(self.current_response)
            self.avg_hist = self.avg_hist*(1-self.homeo_lr) + new_hist*self.homeo_lr

            self.mean_activations = self.mean_activations*(1-self.homeo_lr) + self.current_response*self.homeo_lr
            thresh_update = (self.homeo_target - self.mean_activations) / self.homeo_target
            self.thresholds -= thresh_update * self.homeo_lr
            self.thresholds = self.thresholds.clip(-1,1)

            beta_os = self.current_response.detach()
            self.old_style_mean_fr = self.old_style_mean_fr * (1 - beta_os) + self.current_response * beta_os
            self.old_style_mean_aff = self.old_style_mean_aff * (1 - beta_os) + afferent_delta.detach() * beta_os
            if layer_3:

                gated_response_l3 = self.current_response_l3.detach() * self.homeo_lr * 10
                self.mean_lat_l3 = self.mean_lat_l3 * (1 - gated_response_l3) + torch.relu(delta_l3) * gated_response_l3
                self.delta_mag = (
                    self.delta_mag * (1 - gated_response_l3)
                    + torch.relu(delta_l3) * gated_response_l3
                )

                self.mean_aff_l3 = self.mean_aff_l3 * (1 - gated_response_l3) + torch.relu(afferent_delta_l3) * gated_response_l3
                self.mean_fr_l3 = self.mean_lat_l3 + self.mean_aff_l3
                self.mix_l3 = self.mean_lat_l3.mean() / (self.mean_lat_l3.mean() + self.mean_aff_l3.mean() + 1e-11)
                target_aff_l3 = self.act_target * (1 - self.lat_dom_l3)
                target_lat_l3 = self.act_target * self.lat_dom_l3
                beta_error_l3 = (self.mean_aff_l3.mean() - target_aff_l3) / (target_aff_l3 + 1e-11)
                gain_error_l3 = (self.mean_lat_l3.mean() - target_lat_l3) / (target_lat_l3 + 1e-11)
                self.aff_gain_l3 -= beta_error_l3 * self.homeo_lr
                self.aff_gain_l3 = self.aff_gain_l3.clip(0.1)
                self.lat_gain_l3 -= gain_error_l3 * self.homeo_lr
                self.lat_gain_l3 = self.lat_gain_l3.clip(0.1)

                new_hist = self._activity_histogram(self.current_response_l3)
                self.avg_hist_l3 = self.avg_hist_l3*(1-self.homeo_lr) + new_hist*self.homeo_lr

                self.mean_activations_l3 = self.mean_activations_l3*(1-self.homeo_lr) + self.current_response_l3*self.homeo_lr
                thresh_update = (self.homeo_target - self.mean_activations_l3) / self.homeo_target
                self.thresholds_l3 -= thresh_update * self.homeo_lr / 4
                self.thresholds_l3 = self.thresholds_l3.clip(-1,1)

                beta_os_l3 = self.current_response_l3.detach()
                self.old_style_mean_fr_l3 = self.old_style_mean_fr_l3 * (1 - beta_os_l3) + self.current_response_l3 * beta_os_l3
                self.old_style_mean_aff_l3 = self.old_style_mean_aff_l3 * (1 - beta_os_l3) + afferent_delta_l3.detach() * beta_os_l3

                self._update_global_l3_thresholds()


    def hebbian_step(self, layer_3=None):

        if layer_3 is None:
            layer_3 = self._last_forward_layer_3

        self.step(self.afferent_weights, self.current_tiles, self.current_response.view(-1,1,1,1))
        self.step(
            self.lateral_correlations,
            self.current_response,
            self.current_response.view(-1,1,1,1),
        )
        thresh = 0.0
        self.step(self.lateral_correlations_exc, self.current_response, (self.current_response.view(-1,1,1,1) - thresh), unlearning=0e-3)

        if not layer_3:
            return

        self.step(self.afferent_weights_l3, self.current_tiles_l3, self.current_response_l3.view(-1,1,1,1), lr=self.hebbian_lr)
        self.step(
            self.lateral_correlations_l3,
            self.current_response_l3,
            (self.current_response_l3 - self.global_inh_threshold_l3).view(-1,1,1,1),
            lr=self.hebbian_lr,
        )

        self.step(self.lateral_correlations_l4_l3, self.current_response, self.current_response_l3.view(-1,1,1,1), lr=self.hebbian_lr)
        self.step(self.lateral_correlations_exc_l4_l3, self.current_response, 4*self.current_response_l3.view(-1,1,1,1) - thresh, lr=self.hebbian_lr)

        self.step(
            self.lateral_correlations_exc_l3,
            self.current_response_l3,
            (self.current_response_l3 - self.global_exc_threshold_l3).view(-1,1,1,1),
            unlearning=0e-3,
            lr=self.hebbian_lr,
        )

    def _update_global_l3_thresholds(self):
        response = self.current_response_l3.detach()
        global_exc = (self.global_exc_l3 * response).sum([1, 2, 3]).view(response.shape)
        global_inh = (self.global_inh_l3 * response).sum([1, 2, 3]).view(response.shape)
        gated_response = response * self.homeo_lr * 100
        self.rw_global_exc_l3 = (
            self.rw_global_exc_l3 * (1 - gated_response)
            + global_exc * gated_response
        )
        self.rw_global_inh_l3 = (
            self.rw_global_inh_l3 * (1 - gated_response)
            + global_inh * gated_response
        )
        self.rw_global_net_l3 = self.rw_global_exc_l3 - self.rw_global_inh_l3

        exc_error = (self.rw_global_exc_l3 - self.target_rw_global_exc_l3) / self.target_rw_global_exc_l3
        inh_error = (self.rw_global_inh_l3 - self.target_rw_global_inh_l3) / self.target_rw_global_inh_l3
        self.global_exc_threshold_l3 = torch.clamp(
            self.global_exc_threshold_l3 - exc_error * self.homeo_lr / 10,
            min=-1,
            max=0.1,
        )
        self.global_inh_threshold_l3 = torch.clamp(
            self.global_inh_threshold_l3 - inh_error * self.homeo_lr / 10,
            min=-1,
            max=0,
        )

    def _activity_histogram(self, response):
        active = response[response > 0]
        if active.numel() == 0:
            return torch.zeros_like(self.avg_hist)
        if torch.are_deterministic_algorithms_enabled():
            n_bins = self.avg_hist.numel()
            bin_indices = torch.clamp(
                torch.floor(active * n_bins).to(torch.long),
                min=0,
                max=n_bins - 1,
            )
            bin_ids = torch.arange(n_bins, device=active.device)
            return (bin_indices[:, None] == bin_ids[None, :]).sum(0)
        return torch.histc(active, bins=self.avg_hist.numel(), min=0, max=1)

    def step(self, weights, target, response, unlearning=0, lr=None):

        delta = response * target - unlearning
        weights += (self.hebbian_lr if lr is None else lr) * delta
        weights *= weights > 0
        weights /= weights.mean([1,2,3], keepdim=True) + 1e-11


    def get_aff_weights(self):

        aff_weights = self.afferent_weights * self.aff_cutoff
        aff_weights /= aff_weights.sum([1,2,3], keepdim=True) + 1e-11
        return aff_weights

    def get_aff_weights_l3(self):

        aff_weights = self.afferent_weights_l3 * self.aff_cutoff_l3
        aff_weights /= aff_weights.sum([1,2,3], keepdim=True) + 1e-11
        return aff_weights


    def update_interactions(self, layer_3=True):

        self._ensure_l4_sparsity_freeze_state()
        self.inh = self.i_cutoff * self.lateral_correlations * self.l4_inh_sparsity
        self.inh /= self.inh.sum([2,3], keepdim=True) + 1e-11

        if self.microcolumnar:

            self.l_exc = self.le_cutoff * self.lateral_correlations_exc + 1e-11
            self.l_exc /= (self.l_exc).sum([2,3], keepdim=True)

        if not layer_3:
            return

        self.inh_l3 = self.i_cutoff * self.lateral_correlations_l3
        self.inh_l3 /= self.inh_l3.sum([2,3], keepdim=True)

        self.inh_l4_l3 = self.i_cutoff * self.lateral_correlations_l4_l3
        self.inh_l4_l3 /= self.inh_l4_l3.sum([2,3], keepdim=True)

        self.s_exc_l3 = self.se_cutoff * self.loc_lat_sparsity + 1e-11 * self.eye
        self.s_exc_l3 /= self.s_exc_l3.sum([2,3], keepdim=True)

        if self.microcolumnar:

            self.l_exc_l4_l3 = self.le_cutoff * self.lateral_correlations_exc_l4_l3
            self.l_exc_l4_l3 /= self.l_exc_l4_l3.sum([2,3], keepdim=True) + 1e-11

            self.l_exc_l3 = self.le_cutoff * self.lateral_correlations_exc_l3 * self.lat_sparsity + 1e-11
            self.l_exc_l3 /= (self.l_exc_l3).sum([2,3], keepdim=True)


        self.global_exc_l3 = (
            self.lateral_correlations_exc_l3 * self.global_exc_sparsity_l3
            + 1e-11 * self.global_exc_sparsity_l3
        )
        self.global_exc_l3 /= (self.global_exc_l3).sum([1,2,3], keepdim=True)

        self.global_inh_l3 = (
            self.lateral_correlations_l3 * self.global_inh_sparsity_l3
            + 1e-11 * self.global_inh_sparsity_l3
        )
        self.global_inh_l3 /= (self.global_inh_l3).sum([1,2,3], keepdim=True)

    def _weighted_connection_mask(self, weights, sparsity, eligible_mask=None):
        flat = weights.detach().clamp_min(0).reshape(weights.shape[0], -1)
        if eligible_mask is None:
            eligible = torch.ones_like(flat, dtype=torch.bool)
        else:
            eligible = eligible_mask.reshape(weights.shape[0], -1).to(
                device=weights.device,
                dtype=torch.bool,
            )
        eligible_counts = eligible.sum(1)
        keep_counts = torch.round((1 - float(sparsity)) * eligible_counts.float()).long()
        keep_counts = torch.maximum(torch.ones_like(keep_counts), keep_counts)
        keep_counts = torch.minimum(eligible_counts, keep_counts)
        if torch.all(keep_counts == eligible_counts):
            return eligible.view_as(weights)

        mask = torch.zeros_like(flat, dtype=torch.bool)
        masked_flat = flat * eligible
        for row_idx in range(flat.shape[0]):
            keep_count = int(keep_counts[row_idx].item())
            if keep_count == 0:
                continue
            row_eligible = eligible[row_idx]
            if keep_count >= int(eligible_counts[row_idx].item()):
                mask[row_idx] = row_eligible
                continue
            probs = masked_flat[row_idx]
            probs_sum = probs.sum()
            if probs_sum <= 0:
                probs = row_eligible.float()
                probs_sum = probs.sum()
            indices = torch.multinomial(probs / probs_sum, keep_count, replacement=False)
            mask[row_idx, indices] = True
        return mask.view_as(weights)

    def _ensure_l4_sparsity_freeze_state(self):
        if not hasattr(self, 'frozen_l4_inh_sparsity'):
            self.frozen_l4_inh_sparsity = torch.zeros_like(
                self.lateral_correlations,
                dtype=torch.bool,
            )
        if not hasattr(self, 'frozen_l4_sparsity'):
            self.frozen_l4_sparsity = torch.tensor(-1.0, device=self.device)
        if not hasattr(self, 'l4_sparsity_freeze_active'):
            self.l4_sparsity_freeze_active = torch.tensor(False, device=self.device)
        if not hasattr(self, 'l4_inh_sparsity'):
            self.l4_inh_sparsity = 1

    def _ensure_global_sparsity_freeze_state(self):
        if not hasattr(self, 'frozen_global_exc_sparsity_l3'):
            self.frozen_global_exc_sparsity_l3 = torch.zeros_like(
                self.lateral_correlations_exc_l3,
                dtype=torch.bool,
            )
        if not hasattr(self, 'frozen_global_inh_sparsity_l3'):
            self.frozen_global_inh_sparsity_l3 = torch.zeros_like(
                self.lateral_correlations_l3,
                dtype=torch.bool,
            )
        if not hasattr(self, 'frozen_global_sparsity_l3'):
            self.frozen_global_sparsity_l3 = torch.tensor(-1.0, device=self.device)
        if not hasattr(self, 'global_sparsity_freeze_active_l3'):
            self.global_sparsity_freeze_active_l3 = torch.tensor(False, device=self.device)


    def init_indices(self):

        N = self.sheet_size

        num_images = N**2

        batch_indices = torch.arange(num_images).view(num_images, 1, 1, 1)
        # Create a batch dimension for indices
        self.batch_indices = batch_indices.expand(num_images, 1, self.window, self.window)

        # Generate all possible row and column starts
        row_indices = torch.arange(0, N).repeat_interleave(N)
        col_indices = torch.arange(0, N).repeat(N)

        # Expand indices to use for gathering
        row_indices = row_indices.view(num_images, 1, 1).expand(num_images, self.window, self.window)
        col_indices = col_indices.view(num_images, 1, 1).expand(num_images, self.window, self.window)

        # Create range tensors for MxM crops
        range_rows = torch.arange(0, self.window).view(1, self.window, 1).expand(num_images, self.window, self.window)
        range_cols = torch.arange(0, self.window).view(1, 1, self.window).expand(num_images, self.window, self.window)

        # Add start indices and range indices
        self.final_row_indices = (row_indices + range_rows).view(num_images, 1, self.window, self.window).to(self.device)
        self.final_col_indices = (col_indices + range_cols).view(num_images, 1, self.window, self.window).to(self.device)
