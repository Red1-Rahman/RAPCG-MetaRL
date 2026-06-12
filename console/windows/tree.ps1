# tree.ps1 — Python & Jupyter file tree for RAPCG-MetaRL
# Usage: .\console\tree.ps1
# Run from project root: D:\Work\thesis\RAPCG-MetaRL\

function Get-PythonTree {
    param (
        [string]$Path = ".",
        [string]$Indent = "",
        [bool]$RootLevel = $true
    )

    $ExcludeFolders = @('__pycache__', '.github', '.qodo', '.ruff_cache', '.vscode', 'pcg_env')

    $items = Get-ChildItem -Path $Path | Where-Object {
        if ($_.Name -in $ExcludeFolders) { return $false }

        if ($_.PSIsContainer) {
            $hasFiles = Get-ChildItem -Path $_.FullName -Recurse -File | Where-Object {
                $isValidFile = $_.Extension -in '.py', '.ipynb'
                if (-not $isValidFile) { return $false }
                $pathParts = $_.FullName -split '[\\/]'
                foreach ($folder in $ExcludeFolders) {
                    if ($pathParts -contains $folder) { return $false }
                }
                return $true
            }
            return [bool]$hasFiles
        }
        else {
            return ($_.Extension -in '.py', '.ipynb')
        }
    }

    $count = $items.Count
    for ($i = 0; $i -lt $count; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq $count - 1)
        $pointer = if ($isLast) { "└── " } else { "├── " }
        Write-Output "${Indent}${pointer}$($item.Name)"

        if ($item.PSIsContainer) {
            $nextIndent = if ($isLast) { $Indent + "    " } else { $Indent + "│   " }
            Get-PythonTree -Path $item.FullName -Indent $nextIndent -RootLevel $false
        }
    }
}

Get-PythonTree
