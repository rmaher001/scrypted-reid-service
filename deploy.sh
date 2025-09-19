#!/bin/bash
set -e

# Smart Notifier Plugin - Remote Debug Deployment Script
# Default server: 192.168.86.74

DEFAULT_IP="192.168.86.74"
SERVER_IP="${1:-$DEFAULT_IP}"

echo "🚀 Deploying ReID Service Plugin (Python) to Scrypted Server"
echo "Server: $SERVER_IP"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Run this script from the plugin directory."
    exit 1
fi

# Check if plugin name matches
if ! grep -q "@scrypted/reid-service" package.json; then
    echo "❌ Error: This doesn't appear to be the reid-service plugin directory."
    exit 1
fi

echo "📦 Building plugin..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please fix compilation errors."
    exit 1
fi

echo "✅ Build successful!"
echo ""

echo "🔗 Deploying to remote server..."
npx scrypted-deploy-debug $SERVER_IP

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Open Scrypted Management Console: http://$SERVER_IP:10443"
    echo "2. Go to Plugins → ReID Service"
    echo "3. Add as mixin to cameras with object detection"
    echo "4. Configure settings through the Scrypted UI"
    echo ""
    echo "⚠️  Note: Python dependencies will be installed on first run"
    echo ""
else
    echo "❌ Deployment failed. Check authentication with 'npx scrypted login'"
    echo "💡 Make sure you can access: http://$SERVER_IP:10443"
fi