def find_str_indices(target_str_list, str_tokens, start_index, end_index):
    """Locate the shortest [l, r) range in str_tokens[start_index:end_index] whose
    concatenation contains every target string. Returns (-1, -1) if no match."""
    l, r = start_index, start_index + 1
    while r <= end_index:
        current_str = "".join(str_tokens[l:r])
        if all(ts in current_str for ts in target_str_list):
            while l + 1 < r:
                if not all(ts in "".join(str_tokens[l + 1 : r]) for ts in target_str_list):
                    break
                l += 1
            return l, r
        r += 1
    return -1, -1
