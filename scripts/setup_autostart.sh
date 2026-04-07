#!/usr/bin/env bash
# scripts/setup_autostart.sh — Install launchd agents to auto-start trading on login.
# Run once: bash scripts/setup_autostart.sh
set -euo pipefail
FIRM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS"

# ── Paper (swing) trading agent ────────────────────────────────────────────────
cat > "$LAUNCH_AGENTS/com.tradingfirm.paper.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>         <string>com.tradingfirm.paper</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$FIRM_DIR/scripts/run_paper.sh</string>
    </array>
    <key>WorkingDirectory</key> <string>$FIRM_DIR</string>
    <key>RunAtLoad</key>         <true/>
    <key>KeepAlive</key>         <true/>
    <key>StandardOutPath</key>   <string>$FIRM_DIR/logs/launchd_paper.log</string>
    <key>StandardErrorPath</key> <string>$FIRM_DIR/logs/launchd_paper.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OMP_NUM_THREADS</key>    <string>1</string>
        <key>MKL_NUM_THREADS</key>    <string>1</string>
        <key>OPENBLAS_NUM_THREADS</key> <string>1</string>
        <key>PATH</key> <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

# ── Intraday trading agent ─────────────────────────────────────────────────────
cat > "$LAUNCH_AGENTS/com.tradingfirm.intraday.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>         <string>com.tradingfirm.intraday</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$FIRM_DIR/scripts/run_intraday.sh</string>
    </array>
    <key>WorkingDirectory</key> <string>$FIRM_DIR</string>
    <key>RunAtLoad</key>         <true/>
    <key>KeepAlive</key>         <true/>
    <key>StandardOutPath</key>   <string>$FIRM_DIR/logs/launchd_intraday.log</string>
    <key>StandardErrorPath</key> <string>$FIRM_DIR/logs/launchd_intraday.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>OMP_NUM_THREADS</key>    <string>1</string>
        <key>MKL_NUM_THREADS</key>    <string>1</string>
        <key>OPENBLAS_NUM_THREADS</key> <string>1</string>
        <key>PATH</key> <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

# Load them now
launchctl load "$LAUNCH_AGENTS/com.tradingfirm.paper.plist"    2>/dev/null && echo "paper agent loaded" || echo "paper agent: load skipped (already loaded or will activate on next login)"
launchctl load "$LAUNCH_AGENTS/com.tradingfirm.intraday.plist" 2>/dev/null && echo "intraday agent loaded" || echo "intraday agent: load skipped"

echo ""
echo "Autostart configured. Both processes will start automatically after login."
echo "To uninstall: launchctl unload $LAUNCH_AGENTS/com.tradingfirm.*.plist"
