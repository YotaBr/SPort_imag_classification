const input = document.getElementById('imageUpload');
const preview = document.getElementById('preview');
const result = document.getElementById('result');
const spinner = document.getElementById('spinner');
const historyList = document.getElementById('historyList');
const dropZone = document.getElementById('drop-zone');
const loadingMsg = document.getElementById('loadingMsg');
const resetBtn = document.getElementById('resetBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

dropZone.addEventListener('click', () => {
  input.click();
});

input.addEventListener('change', () => {
  if (input.files.length > 0) {
    validateAndHandle(input.files[0]);
  }
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('hover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('hover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('hover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    validateAndHandle(files[0]);
  }
});

function showError(message) {
  result.textContent = message;
  result.style.color = 'red';
}

function validateAndHandle(file) {
  if (!file?.type?.startsWith('image/')) {
    showError("Please upload a valid image file.");
    return;
  }
  
  const img = new Image();
  img.onload = function () {
    if (img.width < 224 || img.height < 224) {
      showError('Image too small. Minimum size is 224x224 pixels.');
      input.value = '';
      return;
    }
    handleUpload(file);
  };
  img.src = URL.createObjectURL(file);
}

function handleUpload(file) {
  result.textContent = '';
  result.style.color = '';
  preview.style.display = 'none';
  preview.src = '';
  spinner.style.display = 'block';
  loadingMsg.style.display = 'block';
  resetBtn.style.display = 'none';

  const formData = new FormData();
  formData.append('image', file);

  const reader = new FileReader();
  reader.onload = function (e) {
    preview.src = e.target.result;
    preview.style.display = 'block';
    preview.style.opacity = 0;
    requestAnimationFrame(() => {
      preview.style.transition = 'opacity 0.5s';
      preview.style.opacity = 1;
    });
  };
  reader.readAsDataURL(file);

  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(res => res.json()) 
    .then(data => {
      if (data.status === 'success') {
        const confidence = (data.confidence * 100).toFixed(2);
        result.textContent = `Prediction: ${data.class} (Confidence: ${confidence}%)`;
        result.style.color = 'green';

        const entry = document.createElement('li');
        entry.textContent = `${data.class} (${confidence}%)`;
        historyList.prepend(entry);
      } else {
        showError(`Error: ${data.error}`);
      }
    })
    .catch(err => {
      console.error(err);
      showError('An error occurred.');
    })
    .finally(() => {
      spinner.style.display = 'none';
      loadingMsg.style.display = 'none';
      resetBtn.style.display = 'inline-block';
    });
}

resetBtn.addEventListener('click', () => {
  preview.src = '';
  preview.style.opacity = 0;
  preview.style.display = 'none';
  result.textContent = '';
  result.style.color = '';
  input.value = '';
  resetBtn.style.display = 'none';
  loadingMsg.style.display = 'none';
  spinner.style.display = 'none';
});

clearHistoryBtn.addEventListener('click', () => {
  while (historyList.firstChild) {
    historyList.removeChild(historyList.firstChild);
  }
});