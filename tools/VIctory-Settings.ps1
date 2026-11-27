# VIctory settings: a small window to change the runtime's picture and speed options (quality.toml).
# It does not touch the game: it reads and writes quality.toml, which the runtime (and Play-Fork.ps1 -Quality) turn into
# VICTORY_* settings at the next start. Windows PowerShell 5.1 + WinForms only; nothing to install.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\VIctory-Settings.ps1 [-File <quality.toml>] [-Exe <victory.exe>]
#                                                                                 [-Play <Play-Fork.ps1>] [-Fork v7] [-SelfTest]
# The option list comes from `victory.exe quality --list --json` when the exe has that command, else from
# quality-catalog.json next to this script (tools\gen-quality-catalog.py makes it from the source).
param(
    [string]$File = (Join-Path $env:LOCALAPPDATA 'victory\quality.toml'),
    [string]$Exe = (Join-Path $PSScriptRoot '..\target\release\victory.exe'),
    [string]$Play = (Join-Path $PSScriptRoot '..\Play-VIctory.cmd'),
    [string]$Fork = 'v7',
    [switch]$SelfTest
)
$ErrorActionPreference = 'Stop'

function Get-Catalog {
    if (Test-Path -LiteralPath $Exe) {
        try {
            $text = (& $Exe quality --list --json) -join "`n"
            if ($text.TrimStart().StartsWith('{')) { return $text | ConvertFrom-Json }
        } catch { }
    }
    $json = Join-Path $PSScriptRoot 'quality-catalog.json'
    if (-not (Test-Path -LiteralPath $json)) { throw "No option list: $Exe has no 'quality --list --json' and $json is missing." }
    return Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
}

# --- quality.toml: `preset = "x"` and, under [options], `key = value` lines.
function Read-Quality([string]$Path) {
    $r = @{ preset = 'balanced'; options = @{} }
    if (-not (Test-Path -LiteralPath $Path)) { return $r }
    $section = ''
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        $t = $line.Trim()
        if ($t -eq '' -or $t.StartsWith('#')) { continue }
        if ($t -match '^\[(.+)\]$') { $section = $Matches[1]; continue }
        if ($t -match '^([A-Za-z0-9_]+)\s*=\s*(.*)$') {
            $k = $Matches[1]; $v = $Matches[2].Trim()
            if ($v.StartsWith('"')) { $v = $v.Substring(1, $v.LastIndexOf('"') - 1) }
            else { $v = ($v -replace '\s+#.*$', '').Trim() }
            if ($section -eq '' -and $k -eq 'preset') { $r.preset = $v }
            elseif ($section -eq 'options') { $r.options[$k] = $v }
        }
    }
    return $r
}

function Format-Quality([string]$Preset, $Options) {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.AppendLine('# VIctory quality settings (written by tools\VIctory-Settings.ps1; victory quality --list explains each key).')
    [void]$sb.AppendLine('# Environment variables that are already set win over this file.')
    [void]$sb.AppendLine("preset = `"$Preset`"")
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine('[options]')
    foreach ($k in ($Options.Keys | Sort-Object)) {
        $v = [string]$Options[$k]
        $d = 0.0
        if ($v -eq 'true' -or $v -eq 'false' -or [double]::TryParse($v, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$d)) {
            [void]$sb.AppendLine("$k = $v")
        } else {
            [void]$sb.AppendLine("$k = `"$v`"")
        }
    }
    return $sb.ToString()
}

# The screens this PC has, as values for `monitor`: primary, DISPLAYn (Windows' name of the output) and each monitor's own
# model name (the runtime matches a part of it). The runtime's log lists the same at start.
function Get-MonitorChoices {
    $items = @('primary')
    try {
        Add-Type -AssemblyName System.Windows.Forms
        foreach ($s in [System.Windows.Forms.Screen]::AllScreens) { $items += $s.DeviceName.Replace('\.\', '') }
    } catch { }
    try {
        foreach ($m in (Get-CimInstance -Namespace root\wmi -ClassName WmiMonitorID -ErrorAction Stop)) {
            $n = -join ($m.UserFriendlyName | Where-Object { $_ -ne 0 } | ForEach-Object { [char]$_ })
            if ($n) { $items += $n }
        }
    } catch { }
    return @($items | Select-Object -Unique)
}

# What the user typed in an option's box, checked against its kind; returns an error text or $null.
function Test-Value($o, [string]$v) {
    $v = $v.Trim()
    if ($v -eq '' -or $v -eq '(preset)' -or $v -eq 'default') { return $null }
    switch ($o.kind) {
        'bool' { if ($v -notin 'on', 'off', 'true', 'false', '1', '0') { return "$($o.key): use on, off or default" } }
        'num' {
            $d = 0.0
            if (-not [double]::TryParse($v, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$d)) { return "$($o.key): '$v' is not a number" }
            if ($d -lt $o.min -or $d -gt $o.max) { return "$($o.key): $v is outside $($o.min) to $($o.max)" }
        }
        'choice' { if ($o.words -notcontains $v) { return "$($o.key): use one of $($o.words -join ', ')" } }
    }
    return $null
}

function ConvertTo-FileValue($o, [string]$v) {
    $v = $v.Trim()
    if ($o.kind -eq 'bool') {
        if ($v -in 'on', '1') { return 'true' }
        if ($v -in 'off', '0') { return 'false' }
    }
    return $v
}

function ConvertFrom-FileValue($o, [string]$v) {
    if ($o.kind -eq 'bool') {
        if ($v -eq 'true') { return 'on' }
        if ($v -eq 'false') { return 'off' }
    }
    return $v
}

# The preset's value of an option as the window shows it.
function Show-PresetValue($o, [int]$index) {
    $p = [string]$o.presets[$index]
    if ($p -eq '') { return 'unset' }
    if ($o.kind -eq 'bool') { if ($p -in '1', 'true', 'on') { return 'on' } else { return 'off' } }
    return $p
}

$catalog = Get-Catalog
$presets = @($catalog.presets)

if ($SelfTest) {
    $q = Read-Quality $File
    $o0 = $catalog.options[0]
    $text = Format-Quality 'quality' @{ display_mode = '2560x1440'; ambient = 1.0; nan_scrub = 'true' }
    $tmp = [IO.Path]::GetTempFileName()
    Set-Content -LiteralPath $tmp -Value $text -Encoding UTF8
    $back = Read-Quality $tmp
    Remove-Item $tmp
    if ($back.preset -ne 'quality' -or $back.options['display_mode'] -ne '2560x1440') { throw 'self-test: round trip failed' }
    Write-Host "OK: $($catalog.options.Count) options in $($catalog.groups.Count) groups, presets $($presets -join '/'); file $File preset '$($q.preset)'"
    return
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$state = Read-Quality $File
$form = New-Object System.Windows.Forms.Form
$form.Text = 'VIctory settings'
$form.Size = New-Object System.Drawing.Size(900, 720)
$form.MinimumSize = New-Object System.Drawing.Size(900, 520)
$form.StartPosition = 'CenterScreen'
$form.Font = New-Object System.Drawing.Font('Segoe UI', 9)

$top = New-Object System.Windows.Forms.Panel
$top.Dock = 'Top'; $top.Height = 78
$form.Controls.Add($top)
$lbl = New-Object System.Windows.Forms.Label
$lbl.Text = 'Preset'; $lbl.Location = '12,14'; $lbl.AutoSize = $true
$top.Controls.Add($lbl)
$presetBox = New-Object System.Windows.Forms.ComboBox
$presetBox.DropDownStyle = 'DropDownList'; $presetBox.Location = '70,10'; $presetBox.Width = 150
[void]$presetBox.Items.AddRange($presets)
$presetBox.SelectedItem = $state.preset
if ($presetBox.SelectedIndex -lt 0) { $presetBox.SelectedItem = 'balanced' }
$top.Controls.Add($presetBox)
$hint = New-Object System.Windows.Forms.Label
$hint.Location = '240,8'; $hint.Size = New-Object System.Drawing.Size(640, 40)
$hint.Text = "performance = the game's Series S profile at 1080p, fastest. balanced = Series X profile at 1080p with the speed set. quality = 1440p, correct bodies, sharper textures. ultra = 2160p, no texture cap. A box left at (preset) follows the preset; anything else overrides it."
$top.Controls.Add($hint)
$fileLabel = New-Object System.Windows.Forms.Label
$fileLabel.Location = '12,52'; $fileLabel.AutoSize = $true
$fileLabel.Text = "File: $File"
$top.Controls.Add($fileLabel)

$tabs = New-Object System.Windows.Forms.TabControl
$tabs.Dock = 'Fill'
$form.Controls.Add($tabs)

$bottom = New-Object System.Windows.Forms.Panel
$bottom.Dock = 'Bottom'; $bottom.Height = 52
$form.Controls.Add($bottom)
$tabs.BringToFront()

$rows = @{}   # key -> @{ o; box; presetLabel }
$grey = [System.Drawing.Color]::FromArgb(90, 90, 90)
foreach ($g in $catalog.groups) {
    $page = New-Object System.Windows.Forms.TabPage
    $page.Text = $g
    $page.AutoScroll = $true
    [void]$tabs.TabPages.Add($page)
    $y = 8
    foreach ($o in ($catalog.options | Where-Object { $_.group -eq $g })) {
        $enabled = (-not $o.needs_hooks) -or $catalog.hooks_built
        $name = New-Object System.Windows.Forms.Label
        $name.Text = $o.key; $name.Font = New-Object System.Drawing.Font('Segoe UI', 9, [System.Drawing.FontStyle]::Bold)
        $name.Location = New-Object System.Drawing.Point(8, ($y + 3)); $name.Size = New-Object System.Drawing.Size(180, 20)
        $page.Controls.Add($name)
        $box = New-Object System.Windows.Forms.ComboBox
        $box.Location = New-Object System.Drawing.Point(190, $y); $box.Width = 170
        [void]$box.Items.Add('(preset)')
        switch ($o.kind) {
            'bool' { [void]$box.Items.AddRange(@('on', 'off', 'default')); $box.DropDownStyle = 'DropDownList' }
            'choice' { [void]$box.Items.AddRange(@($o.words)); [void]$box.Items.Add('default'); $box.DropDownStyle = 'DropDownList' }
            default {
                [void]$box.Items.Add('default')
                foreach ($p in ($o.presets | Where-Object { $_ -ne '' } | Select-Object -Unique)) { [void]$box.Items.Add($p) }
                if ($o.key -eq 'monitor') { foreach ($m in (Get-MonitorChoices)) { [void]$box.Items.Add($m) } }
                $box.DropDownStyle = 'DropDown'
            }
        }
        $cur = $state.options[$o.key]
        if ($null -eq $cur) { $box.Text = '(preset)' } else { $box.Text = (ConvertFrom-FileValue $o ([string]$cur)) }
        $box.Enabled = $enabled
        $page.Controls.Add($box)
        $pl = New-Object System.Windows.Forms.Label
        $pl.Location = New-Object System.Drawing.Point(372, ($y + 3)); $pl.Size = New-Object System.Drawing.Size(240, 20)
        $pl.ForeColor = $grey
        $page.Controls.Add($pl)
        $tag = New-Object System.Windows.Forms.Label
        $tag.Location = New-Object System.Drawing.Point(620, ($y + 3)); $tag.Size = New-Object System.Drawing.Size(240, 20)
        if (-not $enabled) { $tag.Text = 'needs the game-settings hooks (not built in)'; $tag.ForeColor = [System.Drawing.Color]::Firebrick }
        elseif ($o.live) { $tag.Text = 'can also change live: F8 in game'; $tag.ForeColor = [System.Drawing.Color]::SeaGreen }
        else { $tag.Text = 'applies at next start'; $tag.ForeColor = $grey }
        $page.Controls.Add($tag)
        $ex = New-Object System.Windows.Forms.Label
        $ex.Text = "$($o.help)  Cost: $($o.cost).  [$($o.env)]"
        $ex.ForeColor = $grey
        $ex.Location = New-Object System.Drawing.Point(24, ($y + 26)); $ex.Size = New-Object System.Drawing.Size(820, 34)
        $page.Controls.Add($ex)
        $rows[$o.key] = @{ o = $o; box = $box; presetLabel = $pl }
        $y += 66
    }
}

function Update-PresetLabels {
    $i = $presetBox.SelectedIndex
    foreach ($r in $rows.Values) { $r.presetLabel.Text = 'preset: ' + (Show-PresetValue $r.o $i) }
}
$presetBox.Add_SelectedIndexChanged({ Update-PresetLabels })
Update-PresetLabels

function Get-Overrides {
    $errors = @(); $opts = @{}
    foreach ($k in $rows.Keys) {
        $r = $rows[$k]
        $v = $r.box.Text.Trim()
        if ($v -eq '' -or $v -eq '(preset)') { continue }
        $e = Test-Value $r.o $v
        if ($e) { $errors += $e; continue }
        $opts[$k] = ConvertTo-FileValue $r.o $v
    }
    return @{ errors = $errors; options = $opts }
}

function Save-File {
    $res = Get-Overrides
    if ($res.errors.Count -gt 0) {
        [void][System.Windows.Forms.MessageBox]::Show(($res.errors -join "`n"), 'Not saved', 'OK', 'Warning')
        return $false
    }
    $dir = Split-Path -Parent $File
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { [void](New-Item -ItemType Directory -Path $dir -Force) }
    Set-Content -LiteralPath $File -Value (Format-Quality $presetBox.SelectedItem $res.options) -Encoding UTF8
    return $true
}

function New-Button([string]$text, [int]$x, [int]$w, [scriptblock]$click) {
    $b = New-Object System.Windows.Forms.Button
    $b.Text = $text; $b.Location = New-Object System.Drawing.Point($x, 10); $b.Size = New-Object System.Drawing.Size($w, 32)
    $b.Add_Click($click)
    $bottom.Controls.Add($b)
    return $b
}

[void](New-Button 'Reset overrides' 12 120 {
    foreach ($r in $rows.Values) { if ($r.box.Enabled) { $r.box.Text = '(preset)' } }
})
[void](New-Button 'Show variables' 142 120 {
    if (-not (Test-Path -LiteralPath $Exe)) { [void][System.Windows.Forms.MessageBox]::Show("No $Exe to ask.", 'Show variables'); return }
    if (-not (Save-File)) { return }
    try {
        $out = (& $Exe quality --file $File --env) -join "`n"
        [void][System.Windows.Forms.MessageBox]::Show($out, 'VICTORY_* set by this file (a variable already set wins)')
    } catch { [void][System.Windows.Forms.MessageBox]::Show("$_", 'Show variables') }
})
[void](New-Button 'Save' 560 100 {
    if (Save-File) { $fileLabel.Text = "File: $File  (saved $(Get-Date -Format 'HH:mm:ss'))" }
})
[void](New-Button 'Save and play' 670 130 {
    if (-not (Save-File)) { return }
    if (-not (Test-Path -LiteralPath $Play)) { [void][System.Windows.Forms.MessageBox]::Show("Saved, but $Play was not found.", 'Save and play'); return }
    $args = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$Play`"", '-Fork', $Fork, '-Quality', "`"$File`"")
    Start-Process -FilePath 'powershell.exe' -ArgumentList $args
    $form.Close()
})
[void](New-Button 'Close' 810 70 { $form.Close() })

[void]$form.ShowDialog()
