/* Presentation only. Forms submit their original fields to the original routes. */
(() => {
    'use strict';
    document.querySelectorAll('form[data-k-selection]').forEach(form => {
        const method = form.querySelector('[name="method"]');
        const k = form.querySelector('[name="k"]');
        const update = () => { k.disabled = method.value !== 'manual'; };
        method.addEventListener('change', update);
        window.addEventListener('pageshow', update);
        update();
    });
    document.querySelectorAll('form[method="POST"], form[method="post"]').forEach(form => {
        const status = document.createElement('p');
        status.className = 'form-status';
        status.setAttribute('role', 'status');
        status.setAttribute('aria-live', 'polite');
        status.textContent = 'Running analysis. Results will appear here when ready.';
        form.append(status);
        form.addEventListener('submit', event => {
            if (form.getAttribute('aria-busy') === 'true') { event.preventDefault(); return; }
            form.setAttribute('aria-busy', 'true');
            form.querySelectorAll('button[type="submit"]').forEach(button => {
                button.dataset.originalText = button.textContent;
                button.textContent = 'Running analysis…';
                button.disabled = true;
            });
        });
    });
    window.addEventListener('pageshow', () => {
        document.querySelectorAll('form[aria-busy]').forEach(form => form.removeAttribute('aria-busy'));
        document.querySelectorAll('button[data-original-text]').forEach(button => {
            button.textContent = button.dataset.originalText; button.disabled = false;
        });
    });
    const plots = document.querySelectorAll('main img[src*="plots/"]');
    if (!plots.length || !('HTMLDialogElement' in window)) return;
    const dialog = document.createElement('dialog');
    dialog.className = 'chart-dialog';
    dialog.setAttribute('aria-labelledby', 'chart-dialog-title');
    const header = document.createElement('div'); header.className = 'dialog-header';
    const title = document.createElement('h2'); title.id = 'chart-dialog-title';
    const close = document.createElement('button'); close.type = 'button'; close.className = 'dialog-close'; close.textContent = 'Close'; close.setAttribute('aria-label', 'Close chart');
    const image = document.createElement('img');
    header.append(title, close); dialog.append(header, image); document.body.append(dialog);
    let trigger;
    close.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener('close', () => trigger?.focus());
    plots.forEach(plot => {
        const frame = document.createElement('div'); frame.className = 'chart-frame'; plot.before(frame); frame.append(plot);
        const expand = document.createElement('button'); expand.className = 'chart-expand'; expand.type = 'button'; expand.textContent = 'Expand chart ↗';
        const label = (plot.alt || 'Analysis chart').replace(/\.png$/i, '').replaceAll('_', ' ');
        expand.setAttribute('aria-label', `Expand ${label}`);
        expand.addEventListener('click', () => { trigger = expand; title.textContent = label; image.src = plot.src; image.alt = plot.alt; dialog.showModal(); });
        frame.append(expand);
    });
})();
