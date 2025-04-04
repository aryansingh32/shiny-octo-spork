// Build debug script
const fs = require('fs');
const path = require('path');

// Log the current working directory
console.log('Current working directory:', process.cwd());

// Check if dist directory exists
const distPath = path.join(process.cwd(), 'dist');
console.log('Dist path:', distPath);
console.log('Dist exists:', fs.existsSync(distPath));

if (fs.existsSync(distPath)) {
  // List contents of dist
  const distContents = fs.readdirSync(distPath);
  console.log('Dist contents:', distContents);
}

// Check if build directory exists
const buildPath = path.join(process.cwd(), 'build');
console.log('Build path:', buildPath);
console.log('Build exists:', fs.existsSync(buildPath));

if (fs.existsSync(buildPath)) {
  // List contents of build
  const buildContents = fs.readdirSync(buildPath);
  console.log('Build contents:', buildContents);
}

// Check the public directory
const publicPath = path.join(process.cwd(), 'public');
console.log('Public path:', publicPath);
console.log('Public exists:', fs.existsSync(publicPath));

if (fs.existsSync(publicPath)) {
  // List contents of public
  const publicContents = fs.readdirSync(publicPath);
  console.log('Public contents:', publicContents);
} 