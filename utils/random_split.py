import torch
import math
import random

def random_split_patches(tensor, future_P_N, patch_len, separate_ratio):
    B, S, C = tensor.shape
    total_patches = S // patch_len

    if future_P_N > total_patches:
        raise ValueError("future_Patch_Num is bigger than Total_Patch.")

    if separate_ratio == 0:
        forecast_indices = sorted(random.sample(range(total_patches), future_P_N))
    else:
        num_groups = max(1, round(1 / separate_ratio))
        base_size = future_P_N // num_groups
        remainder = future_P_N % num_groups

        # group_sizes = [base_size, ..., base_size] + distribute remainder
        group_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_groups)]
        assert sum(group_sizes) == future_P_N

        used = set()
        selected_groups = []
        available_starts = list(range(total_patches))
        random.shuffle(available_starts)

        for group_len in group_sizes:
            found = False
            for start in available_starts:
                group = list(range(start, start + group_len))
                if group[-1] >= total_patches:
                    continue
                if any(i in used for i in group):
                    continue
                selected_groups.append(group)
                used.update(group)
                found = True
                break
            if not found:
                raise RuntimeError("Not enough non-overlapping groups could be selected.")

        forecast_indices = sorted([i for group in selected_groups for i in group])

    input_indices = sorted([i for i in range(total_patches) if i not in forecast_indices])

    def get_sequence_indices(indices):
        return [torch.arange(i * patch_len, (i + 1) * patch_len) for i in indices]

    forecast_seq_indices = torch.cat(get_sequence_indices(forecast_indices)).sort().values
    input_seq_indices = torch.cat(get_sequence_indices(input_indices)).sort().values

    input_tensor = tensor[:, input_seq_indices, :]
    target_tensor = tensor[:, forecast_seq_indices, :]

    indices = (input_indices, forecast_indices)
    return input_tensor, target_tensor, indices
