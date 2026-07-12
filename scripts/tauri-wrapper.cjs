const { spawnSync } = require('child_process');
const path = require('path');

const args = process.argv.slice(2);
const isRelease = args.includes('build');

console.log(`\x1b[36m[Sidekick Build] Pre-building whisper_worker (${isRelease ? 'release' : 'debug'})...\x1b[0m`);

const cargoArgs = ['build', '-p', 'whisper_worker'];
if (isRelease) {
  cargoArgs.push('--release');
}

const buildResult = spawnSync('cargo', cargoArgs, {
  cwd: path.join(__dirname, '../src-tauri'),
  stdio: 'inherit'
});

if (buildResult.status !== 0) {
  console.error('\x1b[31m[Sidekick Build] Failed to build whisper_worker\x1b[0m');
  process.exit(buildResult.status || 1);
}

console.log('\x1b[32m[Sidekick Build] whisper_worker built successfully. Starting Tauri...\x1b[0m');

// Run the original tauri CLI with all passed arguments
const tauriResult = spawnSync('npx', ['tauri', ...args], {
  stdio: 'inherit',
  shell: true
});

process.exit(tauriResult.status || 0);
