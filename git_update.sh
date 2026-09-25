#!/bin/bash

# Git Update Script with Enhanced Information

# Get current user and timestamp details
USER=$(whoami)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
DATE_SHORT=$(date +"%Y-%m-%d")
TIME_ONLY=$(date +"%H:%M:%S")
DAY_OF_WEEK=$(date +"%A")
WEEK_NUMBER=$(date +"%V")
MONTH_NAME=$(date +"%B")
YEAR=$(date +"%Y")
TIMEZONE=$(date +"%Z")
EPOCH_TIME=$(date +%s)

# Git information
BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
REPO_NAME=$(basename -s .git $(git config --get remote.origin.url 2>/dev/null) 2>/dev/null || echo "repository")
COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null || echo "N/A")

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Pull latest changes first
echo "🔄 Pulling latest changes from origin/$BRANCH..."
if ! git pull origin "$BRANCH"; then
    echo "⚠️  Warning: Pull encountered issues. Continuing with update..."
fi

# Get status before adding
echo -e "\n📊 Current Status:"
git status --short

# Get changed files information
CHANGED_FILES=$(git status --porcelain | wc -l)
if [ "$CHANGED_FILES" -eq 0 ]; then
    echo -e "\n✅ No changes to commit."
    exit 0
fi

# Show detailed file changes
echo -e "\n📁 Changed Files ($CHANGED_FILES):"
git status --porcelain

echo -e "\n📦 Staging changes..."
git add .

# Create comprehensive commit message
COMMIT_MSG="Update by $USER on $DATE_SHORT at $TIME_ONLY

📅 Date Details:
- Full Date: $DAY_OF_WEEK, $MONTH_NAME $DATE_SHORT $YEAR
- Time: $TIME_ONLY ($TIMEZONE)
- Week: $WEEK_NUMBER
- Epoch: $EPOCH_TIME

👤 User: $USER
🌿 Branch: $BRANCH
📁 Repository: $REPO_NAME
📈 Total Commits: $COMMIT_COUNT
🔄 Changed Files: $CHANGED_FILE$

📋 File Changes:"

# Add detailed file list
CHANGED_LIST=$(git status --porcelain)
if [ ! -z "$CHANGED_LIST" ]; then
    COMMIT_MSG="$COMMIT_MSG\n$CHANGED_LIST"
fi

# Add diff summary (optional - uncomment if you want diff stats)
# DIFF_STATS=$(git diff --cached --stat)
# if [ ! -z "$DIFF_STATS" ]; then
#     COMMIT_MSG="$COMMIT_MSG\n\n📊 Diff Summary:\n$DIFF_STATS"
# fi

# Commit with the detailed message
echo -e "\n💾 Committing changes..."
echo -e "Commit message preview:\n"
echo -e "$COMMIT_MSG" | head -25
if [ $(echo "$COMMIT_MSG" | wc -l) -gt 25 ]; then
    echo "... (truncated preview)"
fi

# Use printf to preserve newlines in commit message
printf "%b" "$COMMIT_MSG" | git commit -F -

# Push to remote
echo -e "\n🚀 Pushing to origin/$BRANCH..."
if git push origin "$BRANCH"; then
    echo -e "\n🎉 Successfully updated repository!"
    echo "========================================"
    echo "👤 User:        $USER"
    echo "📅 Date:        $DATE_SHORT"
    echo "⏰ Time:        $TIME_ONLY"
    echo "🌿 Branch:      $BRANCH"
    echo "📁 Repository:  $REPO_NAME"
    echo "🔄 Changes:     $CHANGED_FILES files"
    echo "📈 Commit #:    $((COMMIT_COUNT + 1))"
    echo "========================================"
else
    echo -e "\n❌ Push failed. Please check your remote repository."
    exit 1
fi