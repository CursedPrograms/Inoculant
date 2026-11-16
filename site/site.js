// Inoculant's page: the feature cards, the numbers table and the year
(function () {
  'use strict';
  const features = [
    ['Identity cloak', 'Projected gradient descent with momentum against ArcFace, the recogniser most face swappers read identity with. Each pixel changes by at most the strength you choose.'],
    ['Hide or Decoy', 'Push the face away from you, or pull it toward an AI-generated face, so a swap made from your photo comes out as nobody.'],
    ['Built to last', 'Every step looks through jittered views of the face (EOT), so the cloak survives re-alignment, resizing, blur and JPEG.'],
    ['Antidote', 'JPEG, blur, median, bit depth, shrink-and-enlarge and noise, each with a slider: see how much of the protection is kept.'],
    ['Inspect', 'No original needed: feature squeezing tells whether a picture carries an identity cloak.'],
    ['No Python', 'Inoculant writes ArcFace\'s gradient graph itself, in C++, and runs it on any DirectX 12 GPU through ONNX Runtime and DirectML.']
  ];
  const grid = document.getElementById('features');
  if (grid) {
    for (const [title, text] of features) {
      const card = document.createElement('article');
      const h = document.createElement('h2');
      h.textContent = title;
      const p = document.createElement('p');
      p.textContent = text;
      card.append(h, p);
      grid.appendChild(card);
    }
  }
  const rows = [
    ['', 'ArcFace similarity to the original'],
    ['Protected (was 1.00)', '0.15'],
    ['After JPEG quality 50', '0.17'],
    ['After shrinking to 25% and back', '0.18'],
    ['After all six purifiers in a row', '0.38'],
    ['Same person from about', '0.40']
  ];
  const table = document.getElementById('numbers');
  if (table) {
    rows.forEach(function (r, i) {
      const tr = document.createElement('tr');
      for (const cell of r) {
        const c = document.createElement(i === 0 ? 'th' : 'td');
        c.textContent = cell;
        tr.appendChild(c);
      }
      table.appendChild(tr);
    });
  }
  const year = document.getElementById('year');
  if (year) year.textContent = String(Math.max(2026, new Date().getFullYear()));
})();
