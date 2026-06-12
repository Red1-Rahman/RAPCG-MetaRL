#!/usr/bin/env bash
# tree.sh — Python & Jupyter file tree for RAPCG-MetaRL
# Usage: bash console/tree.sh
# Run from project root: ~/Work/thesis/RAPCG-MetaRL/

EXCLUDE_DIRS="__pycache__|.github|.qodo|.ruff_cache|.vscode|pcg_env"

print_tree() {
    local path="${1:-.}"
    local indent="${2:-}"
    local is_root="${3:-true}"

    # Collect matching items: dirs with .py/.ipynb inside, and .py/.ipynb files
    local items=()
    while IFS= read -r item; do
        local name
        name=$(basename "$item")

        # Skip excluded dirs at any depth
        if [[ -d "$item" ]]; then
            if echo "$name" | grep -qE "^(${EXCLUDE_DIRS})$"; then
                continue
            fi
            # Only include dirs that contain .py or .ipynb files (excluding hidden dirs)
            if find "$item" -type f \( -name "*.py" -o -name "*.ipynb" \) \
                | grep -qvE "/(${EXCLUDE_DIRS})/"; then
                items+=("$item")
            fi
        else
            # Root level: include .py and .ipynb files
            # Non-root: same
            if [[ "$name" == *.py || "$name" == *.ipynb ]]; then
                items+=("$item")
            fi
        fi
    done < <(find "$path" -maxdepth 1 -mindepth 1 | sort)

    local count=${#items[@]}
    for (( i=0; i<count; i++ )); do
        local item="${items[$i]}"
        local name
        name=$(basename "$item")
        local is_last=false
        [[ $i -eq $((count - 1)) ]] && is_last=true

        local pointer="├── "
        $is_last && pointer="└── "

        echo "${indent}${pointer}${name}"

        if [[ -d "$item" ]]; then
            local next_indent
            $is_last && next_indent="${indent}    " || next_indent="${indent}│   "
            print_tree "$item" "$next_indent" false
        fi
    done
}

print_tree "." "" true
