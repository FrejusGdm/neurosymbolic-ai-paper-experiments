#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# reinstall-claude-extensions.sh
# Backup, clean, and reinstall all Claude Code marketplaces, plugins, and skills
# ============================================================================

# --- Colors & helpers -------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERR]${NC}   %s\n" "$*"; }
phase() { printf "\n${BOLD}${CYAN}━━━ %s ━━━${NC}\n\n" "$*"; }

# --- Parse flags ------------------------------------------------------------
DRY_RUN=false
SKILLS_ONLY=false
PLUGINS_ONLY=false
BACKUP_ONLY=false
SKIP_CLEAN=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY_RUN=true ;;
    --skills-only)  SKILLS_ONLY=true ;;
    --plugins-only) PLUGINS_ONLY=true ;;
    --backup-only)  BACKUP_ONLY=true ;;
    --skip-clean)   SKIP_CLEAN=true ;;
    --help|-h)
      cat <<'USAGE'
Usage: reinstall-claude-extensions.sh [OPTIONS]

Options:
  --dry-run        Show what would happen without doing anything
  --skills-only    Only reinstall skills (skip marketplaces/plugins)
  --plugins-only   Only reinstall marketplaces and plugins (skip skills)
  --backup-only    Only create backup, then exit
  --skip-clean     Skip the clean phase (install on top of existing)
  -h, --help       Show this help
USAGE
      exit 0
      ;;
    *)
      err "Unknown flag: $arg"
      exit 1
      ;;
  esac
done

if $DRY_RUN; then
  info "DRY RUN mode — no changes will be made"
fi

# --- Paths ------------------------------------------------------------------
CLAUDE_DIR="$HOME/.claude"
AGENTS_DIR="$HOME/.agents"
SKILLS_SRC="$AGENTS_DIR/skills"
SKILLS_LINK="$CLAUDE_DIR/skills"
PLUGINS_DIR="$CLAUDE_DIR/plugins"
BACKUP_BASE="$CLAUDE_DIR/backups"

# ============================================================================
# PHASE 1: BACKUP
# ============================================================================
phase "Phase 1: Backup"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE/$TIMESTAMP"

backup_file() {
  local src="$1"
  local label="$2"
  if [[ -f "$src" ]]; then
    if $DRY_RUN; then
      info "Would backup $label → $BACKUP_DIR/"
    else
      cp "$src" "$BACKUP_DIR/"
      ok "Backed up $label"
    fi
  else
    warn "Not found, skipping: $src"
  fi
}

backup_dir() {
  local src="$1"
  local label="$2"
  if [[ -d "$src" ]]; then
    if $DRY_RUN; then
      info "Would backup $label → $BACKUP_DIR/"
    else
      cp -R "$src" "$BACKUP_DIR/$(basename "$src")"
      ok "Backed up $label"
    fi
  else
    warn "Not found, skipping: $src"
  fi
}

if ! $DRY_RUN; then
  mkdir -p "$BACKUP_DIR"
  ok "Created backup directory: $BACKUP_DIR"
fi

backup_file "$AGENTS_DIR/.skill-lock.json"              ".skill-lock.json"
backup_file "$PLUGINS_DIR/installed_plugins.json"        "installed_plugins.json"
backup_file "$PLUGINS_DIR/known_marketplaces.json"       "known_marketplaces.json"
backup_file "$CLAUDE_DIR/settings.json"                  "settings.json"
backup_dir  "$SKILLS_LINK/convex"                        "convex (manual skill)"
backup_dir  "$SKILLS_LINK/livekit-voice-agents"          "livekit-voice-agents (manual skill)"

# Also backup the .skill file if it exists
backup_file "$SKILLS_LINK/livekit-voice-agents.skill"    "livekit-voice-agents.skill"

if $BACKUP_ONLY; then
  ok "Backup complete. Exiting."
  exit 0
fi

# ============================================================================
# PHASE 2: CLEAN (optional)
# ============================================================================
if ! $SKIP_CLEAN; then
  phase "Phase 2: Clean"

  clean_skills() {
    # Remove symlinks in ~/.claude/skills/ (but preserve manual dirs convex, livekit-voice-agents)
    if [[ -d "$SKILLS_LINK" ]]; then
      if $DRY_RUN; then
        info "Would remove symlinks in $SKILLS_LINK/"
        info "Would remove skill dirs in $SKILLS_SRC/"
      else
        # Remove symlinks only
        find "$SKILLS_LINK" -maxdepth 1 -type l -delete
        ok "Removed skill symlinks from $SKILLS_LINK/"

        # Remove skill directories in agents dir
        if [[ -d "$SKILLS_SRC" ]]; then
          rm -rf "$SKILLS_SRC"
          mkdir -p "$SKILLS_SRC"
          ok "Cleaned $SKILLS_SRC/"
        fi

        # Remove old lock file (will be rebuilt)
        rm -f "$AGENTS_DIR/.skill-lock.json"
        ok "Removed .skill-lock.json"
      fi
    fi
  }

  clean_plugins() {
    if $DRY_RUN; then
      info "Would remove plugin cache in $PLUGINS_DIR/cache/"
      info "Would remove installed_plugins.json"
      info "Would remove known_marketplaces.json"
      info "Would remove marketplace dirs in $PLUGINS_DIR/marketplaces/"
    else
      rm -rf "$PLUGINS_DIR/cache"
      rm -f "$PLUGINS_DIR/installed_plugins.json"
      rm -f "$PLUGINS_DIR/known_marketplaces.json"
      rm -rf "$PLUGINS_DIR/marketplaces"
      ok "Cleaned plugin cache, configs, and marketplaces"
    fi
  }

  if ! $SKILLS_ONLY; then
    clean_plugins
  fi
  if ! $PLUGINS_ONLY; then
    clean_skills
  fi
else
  info "Skipping clean phase (--skip-clean)"
fi

# ============================================================================
# PHASE 3: MARKETPLACES
# ============================================================================
if ! $SKILLS_ONLY; then
  phase "Phase 3: Marketplaces"

  # Marketplace definitions: name|source_type|source_value
  # source_type is "github" (owner/repo) or "git" (full URL)
  MARKETPLACES=(
    "claude-plugins-official|github|anthropics/claude-plugins-official"
    "claude-code-plugins|git|https://github.com/anthropics/claude-code.git"
    "claude-code-workflows|git|https://github.com/wshobson/agents.git"
    "huggingface-skills|github|huggingface/skills"
    "dev-browser-marketplace|github|sawyerhood/dev-browser"
  )

  for entry in "${MARKETPLACES[@]}"; do
    IFS='|' read -r name stype source <<< "$entry"
    if $DRY_RUN; then
      info "Would add marketplace: $name ($source)"
    else
      info "Adding marketplace: $name"
      if claude plugin marketplace add "$source" 2>/dev/null; then
        ok "Added $name"
      else
        warn "Marketplace $name may already exist or failed — continuing"
      fi
    fi
  done

  if ! $DRY_RUN; then
    info "Updating all marketplaces..."
    claude plugin marketplace update 2>/dev/null || warn "Marketplace update had warnings"
    ok "Marketplaces updated"
  fi

# ============================================================================
# PHASE 4: PLUGINS
# ============================================================================
  phase "Phase 4: Plugins"

  PLUGINS=(
    "feature-dev@claude-code-plugins"
    "code-documentation@claude-code-workflows"
    "database-design@claude-code-workflows"
    "data-validation-suite@claude-code-workflows"
    "backend-development@claude-code-workflows"
    "huggingface-skills@claude-plugins-official"
    "dev-browser@dev-browser-marketplace"
  )

  for plugin in "${PLUGINS[@]}"; do
    if $DRY_RUN; then
      info "Would install + enable plugin: $plugin"
    else
      info "Installing $plugin"
      if claude plugin install "$plugin" 2>/dev/null; then
        ok "Installed $plugin"
      else
        warn "Install may have failed for $plugin — continuing"
      fi

      info "Enabling $plugin"
      if claude plugin enable "$plugin" 2>/dev/null; then
        ok "Enabled $plugin"
      else
        warn "Enable may have failed for $plugin — continuing"
      fi
    fi
  done
fi

# ============================================================================
# PHASE 5: SKILLS (git clone + copy)
# ============================================================================
if ! $PLUGINS_ONLY; then
  phase "Phase 5: Skills"

  mkdir -p "$SKILLS_SRC" "$SKILLS_LINK"

  TMPDIR_BASE="/tmp/claude-skill-reinstall-$$"
  mkdir -p "$TMPDIR_BASE"
  trap 'rm -rf "$TMPDIR_BASE"' EXIT

  # Track which repos have been cloned (file-based, bash 3.x compatible)
  CLONED_LIST="$TMPDIR_BASE/.cloned"
  FAILED_LIST="$TMPDIR_BASE/.failed"
  touch "$CLONED_LIST" "$FAILED_LIST"

  # Each skill entry: skill_name|repo_slug|clone_url|skill_path_in_repo
  # skill_path_in_repo is the path to SKILL.md; the skill folder is its parent dir
  SKILLS=(
    "skill-creator|anthropics/skills|https://github.com/anthropics/skills.git|skills/skill-creator/SKILL.md"
    "frontend-design|anthropics/skills|https://github.com/anthropics/skills.git|skills/frontend-design/SKILL.md"
    "native-data-fetching|expo/skills|https://github.com/expo/skills.git|plugins/expo-app-design/skills/native-data-fetching/SKILL.md"
    "upgrading-expo|expo/skills|https://github.com/expo/skills.git|plugins/upgrading-expo/skills/upgrading-expo/SKILL.md"
    "expo-deployment|expo/skills|https://github.com/expo/skills.git|plugins/expo-deployment/skills/expo-deployment/SKILL.md"
    "expo-cicd-workflows|expo/skills|https://github.com/expo/skills.git|plugins/expo-deployment/skills/expo-cicd-workflows/SKILL.md"
    "use-dom|expo/skills|https://github.com/expo/skills.git|plugins/expo-app-design/skills/use-dom/SKILL.md"
    "expo-tailwind-setup|expo/skills|https://github.com/expo/skills.git|plugins/expo-app-design/skills/expo-tailwind-setup/SKILL.md"
    "expo-dev-client|expo/skills|https://github.com/expo/skills.git|plugins/expo-app-design/skills/expo-dev-client/SKILL.md"
    "expo-api-routes|expo/skills|https://github.com/expo/skills.git|plugins/expo-app-design/skills/expo-api-routes/SKILL.md"
    "building-native-ui|expo/skills|https://github.com/expo/skills.git|plugins/expo-app-design/skills/building-native-ui/SKILL.md"
    "vercel-react-native-skills|vercel-labs/agent-skills|https://github.com/vercel-labs/agent-skills.git|skills/react-native-skills/SKILL.md"
    "web-design-guidelines|vercel-labs/agent-skills|https://github.com/vercel-labs/agent-skills.git|skills/web-design-guidelines/SKILL.md"
    "vercel-composition-patterns|vercel-labs/agent-skills|https://github.com/vercel-labs/agent-skills.git|skills/composition-patterns/SKILL.md"
    "vercel-react-best-practices|vercel-labs/agent-skills|https://github.com/vercel-labs/agent-skills.git|skills/react-best-practices/SKILL.md"
    "find-skills|vercel-labs/skills|https://github.com/vercel-labs/skills.git|skills/find-skills/SKILL.md"
    "react-native-best-practices|callstackincubator/agent-skills|https://github.com/callstackincubator/agent-skills.git|skills/react-native-best-practices/SKILL.md"
    "remotion-best-practices|remotion-dev/skills|https://github.com/remotion-dev/skills.git|skills/remotion/SKILL.md"
    "convex-backend|cloudai-x/claude-workflow|https://github.com/cloudai-x/claude-workflow.git|skills/convex-backend/SKILL.md"
    "typescript-advanced-types|wshobson/agents|https://github.com/wshobson/agents.git|plugins/javascript-typescript/skills/typescript-advanced-types/SKILL.md"
    "better-auth-best-practices|better-auth/skills|https://github.com/better-auth/skills.git|better-auth/best-practices/SKILL.md"
    "polymarket|2025emma/vibe-coding-cn|https://github.com/2025emma/vibe-coding-cn.git|i18n/zh/skills/polymarket/SKILL.md"
    "programmatic-seo|coreyhaines31/marketingskills|https://github.com/coreyhaines31/marketingskills.git|skills/programmatic-seo/SKILL.md"
  )

  clone_repo() {
    local slug="$1"
    local url="$2"
    local safe_name="${slug//\//_}"
    local dest="$TMPDIR_BASE/$safe_name"

    # Already cloned?
    if grep -qxF "$slug" "$CLONED_LIST" 2>/dev/null; then
      return 0
    fi
    # Already failed?
    if grep -qxF "$slug" "$FAILED_LIST" 2>/dev/null; then
      return 1
    fi

    if $DRY_RUN; then
      info "Would clone $slug → $dest"
      echo "$slug" >> "$CLONED_LIST"
      return 0
    fi

    info "Cloning $slug..."
    if git clone --depth 1 --quiet "$url" "$dest" 2>/dev/null; then
      ok "Cloned $slug"
      echo "$slug" >> "$CLONED_LIST"
      return 0
    else
      err "Failed to clone $slug — skills from this repo will be skipped"
      echo "$slug" >> "$FAILED_LIST"
      return 1
    fi
  }

  install_skill() {
    local name="$1"
    local slug="$2"
    local url="$3"
    local skill_path="$4"  # path to SKILL.md inside repo

    # Clone if needed
    clone_repo "$slug" "$url" || return 0

    local safe_name="${slug//\//_}"
    local repo_dir="$TMPDIR_BASE/$safe_name"
    local skill_folder
    skill_folder="$(dirname "$skill_path")"

    if $DRY_RUN; then
      info "Would install skill: $name (from $slug:$skill_folder)"
      return 0
    fi

    # Check if clone failed
    if grep -qxF "$slug" "$FAILED_LIST" 2>/dev/null; then
      warn "Skipping $name — repo clone failed"
      return 0
    fi

    local src_dir="$repo_dir/$skill_folder"
    local dest_dir="$SKILLS_SRC/$name"

    if [[ ! -d "$src_dir" ]]; then
      err "Skill folder not found: $src_dir — skipping $name"
      return 0
    fi

    # Copy skill folder contents
    rm -rf "$dest_dir"
    cp -R "$src_dir" "$dest_dir"

    # Create symlink
    ln -sf "../../.agents/skills/$name" "$SKILLS_LINK/$name"

    ok "Installed skill: $name"
  }

  for entry in "${SKILLS[@]}"; do
    IFS='|' read -r name slug url skill_path <<< "$entry"
    install_skill "$name" "$slug" "$url" "$skill_path"
  done

  # --- Rebuild .skill-lock.json ---------------------------------------------
  if ! $DRY_RUN; then
    info "Rebuilding .skill-lock.json..."

    # Build the JSON programmatically
    LOCK_FILE="$AGENTS_DIR/.skill-lock.json"
    NOW=$(date -u +%Y-%m-%dT%H:%M:%S.000Z)

    # Start JSON
    {
      printf '{\n  "version": 3,\n  "skills": {\n'

      first=true
      for entry in "${SKILLS[@]}"; do
        IFS='|' read -r name slug url skill_path <<< "$entry"

        # Skip if the skill wasn't actually installed
        if [[ ! -d "$SKILLS_SRC/$name" ]]; then
          continue
        fi

        # Compute folder hash
        local_hash=""
        if command -v shasum &>/dev/null; then
          local_hash=$(find "$SKILLS_SRC/$name" -type f -print0 | sort -z | xargs -0 shasum -a 1 | shasum -a 1 | cut -d' ' -f1)
        else
          local_hash="0000000000000000000000000000000000000000"
        fi

        if $first; then
          first=false
        else
          printf ',\n'
        fi

        cat <<ENTRY
    "$name": {
      "source": "$slug",
      "sourceType": "github",
      "sourceUrl": "$url",
      "skillPath": "$skill_path",
      "skillFolderHash": "$local_hash",
      "installedAt": "$NOW",
      "updatedAt": "$NOW"
    }
ENTRY
      done

      printf '\n  },\n'
      printf '  "dismissed": {\n    "findSkillsPrompt": true\n  },\n'
      printf '  "lastSelectedAgents": [\n'
      printf '    "amp",\n    "codex",\n    "gemini-cli",\n    "github-copilot",\n'
      printf '    "kimi-cli",\n    "opencode",\n    "claude-code"\n'
      printf '  ]\n'
      printf '}\n'
    } > "$LOCK_FILE"

    ok "Rebuilt .skill-lock.json"
  fi

  # --- Restore manual skills from backup ------------------------------------
  if ! $DRY_RUN && ! $SKIP_CLEAN; then
    info "Restoring manual skills from backup..."

    # convex
    if [[ -d "$BACKUP_DIR/convex" ]]; then
      cp -R "$BACKUP_DIR/convex" "$SKILLS_LINK/convex"
      ok "Restored manual skill: convex"
    else
      warn "convex backup not found — you'll need to reinstall it manually"
    fi

    # livekit-voice-agents
    if [[ -d "$BACKUP_DIR/livekit-voice-agents" ]]; then
      cp -R "$BACKUP_DIR/livekit-voice-agents" "$SKILLS_LINK/livekit-voice-agents"
      ok "Restored manual skill: livekit-voice-agents"
    else
      warn "livekit-voice-agents backup not found — you'll need to reinstall it manually"
    fi

    # Restore the .skill file too
    if [[ -f "$BACKUP_DIR/livekit-voice-agents.skill" ]]; then
      cp "$BACKUP_DIR/livekit-voice-agents.skill" "$SKILLS_LINK/livekit-voice-agents.skill"
      ok "Restored livekit-voice-agents.skill"
    fi
  fi
fi

# ============================================================================
# PHASE 6: MANUAL SKILLS REMINDER
# ============================================================================
phase "Phase 6: Manual Skills"

warn "The following skills were installed manually and cannot be auto-reinstalled:"
printf "  ${YELLOW}convex${NC}             — directory in ~/.claude/skills/convex/\n"
printf "  ${YELLOW}livekit-voice-agents${NC} — from .skill package file\n"
if ! $DRY_RUN && ! $SKIP_CLEAN; then
  info "Both were restored from backup (if available)"
fi
printf "\n"

# ============================================================================
# PHASE 7: VERIFY
# ============================================================================
phase "Phase 7: Verify"

if $DRY_RUN; then
  info "Skipping verification in dry-run mode"
  info "DRY RUN complete — no changes were made"
  exit 0
fi

ERRORS=0

# Check skill symlinks
info "Checking skill symlinks..."
EXPECTED_SYMLINKS=(
  skill-creator frontend-design
  native-data-fetching upgrading-expo expo-deployment expo-cicd-workflows
  use-dom expo-tailwind-setup expo-dev-client expo-api-routes building-native-ui
  vercel-react-native-skills web-design-guidelines vercel-composition-patterns
  vercel-react-best-practices find-skills
  react-native-best-practices remotion-best-practices
  convex-backend typescript-advanced-types
  better-auth-best-practices polymarket programmatic-seo
)

for name in "${EXPECTED_SYMLINKS[@]}"; do
  link="$SKILLS_LINK/$name"
  if [[ -L "$link" ]]; then
    target=$(readlink "$link")
    if [[ -d "$link" ]]; then
      ok "  $name → $target"
    else
      err "  $name → $target (BROKEN)"
      ((ERRORS++))
    fi
  else
    err "  $name — symlink missing"
    ((ERRORS++))
  fi
done

# Check manual skills
for name in convex livekit-voice-agents; do
  if [[ -d "$SKILLS_LINK/$name" ]]; then
    ok "  $name (manual, directory)"
  else
    warn "  $name — not found"
  fi
done

# Check plugins
if ! $SKILLS_ONLY; then
  info "Checking plugins..."
  if command -v claude &>/dev/null; then
    plugin_output=$(claude plugin list 2>/dev/null || true)
    if [[ -n "$plugin_output" ]]; then
      printf "%s\n" "$plugin_output"
    else
      warn "Could not list plugins (claude CLI may not support 'plugin list')"
    fi
  else
    warn "claude CLI not found in PATH"
  fi
fi

# Summary
phase "Summary"

SKILL_COUNT=$(find "$SKILLS_LINK" -maxdepth 1 \( -type l -o -type d \) ! -name skills | wc -l | tr -d ' ')
printf "${BOLD}Skills installed:${NC}  %s (expected: 25)\n" "$SKILL_COUNT"

if ! $SKILLS_ONLY && [[ -f "$PLUGINS_DIR/installed_plugins.json" ]]; then
  PLUGIN_COUNT=$(grep -c '"scope"' "$PLUGINS_DIR/installed_plugins.json" 2>/dev/null || echo 0)
  printf "${BOLD}Plugins installed:${NC} %s (expected: 7)\n" "$PLUGIN_COUNT"
fi

printf "${BOLD}Backup location:${NC}  %s\n" "$BACKUP_DIR"

if [[ $ERRORS -gt 0 ]]; then
  err "$ERRORS verification errors found — check output above"
  exit 1
else
  ok "All checks passed!"
fi
