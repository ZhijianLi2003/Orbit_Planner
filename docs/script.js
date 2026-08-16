const copyButton = document.querySelector("#copy-citation");
const bibtex = document.querySelector("#bibtex");

copyButton?.addEventListener("click", async () => {
  if (!bibtex) return;

  try {
    await navigator.clipboard.writeText(bibtex.textContent ?? "");
    copyButton.textContent = "Copied";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection?.removeAllRanges();
    selection?.addRange(range);
    copyButton.textContent = "Selected — press Ctrl+C";
  }

  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1800);
});
