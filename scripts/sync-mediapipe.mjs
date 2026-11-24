import fs from 'node:fs';
import path from 'node:path';

const srcDir = path.resolve(process.cwd(), 'node_modules/@mediapipe/tasks-vision/wasm');
const dstDir = path.resolve(process.cwd(), 'public/mediapipe');

if (fs.existsSync(srcDir)) {
  fs.mkdirSync(dstDir, { recursive: true });
  for (const file of fs.readdirSync(srcDir)) {
    const srcFile = path.join(srcDir, file);
    const dstFile = path.join(dstDir, file);
    const srcStat = fs.statSync(srcFile);
    if (!srcStat.isFile()) continue;
    if (!fs.existsSync(dstFile) || fs.statSync(dstFile).size !== srcStat.size) {
      fs.copyFileSync(srcFile, dstFile);
    }
  }
}
