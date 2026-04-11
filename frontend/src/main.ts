import './style.css'

interface ApiResponse {
  predicted_count: number;
  security_guards: number;
  high_density_direction: string[];
  directional_density: {
    East: number;
    North: number;
    South: number;
    West: number;
  };
  density_map_image: string;
}

const API_URL = 'http://localhost:8000/predict';

// Selectors
const uploadArea = document.getElementById('upload-area') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const imagePreview = document.getElementById('image-preview') as HTMLImageElement;
const analyzeBtn = document.getElementById('analyze-btn') as HTMLButtonElement;
const btnText = analyzeBtn.querySelector('.btn-text') as HTMLSpanElement;
const btnLoader = document.getElementById('btn-loader') as HTMLDivElement;
const errorMessage = document.getElementById('error-message') as HTMLDivElement;
const resultsSection = document.getElementById('results-section') as HTMLElement;

// Results elements
const resCount = document.getElementById('res-count') as HTMLDivElement;
const resGuards = document.getElementById('res-guards') as HTMLDivElement;
const resHeatmap = document.getElementById('res-heatmap') as HTMLImageElement;
const resHighDir = document.getElementById('res-high-dir') as HTMLSpanElement;
const barchart = document.getElementById('barchart') as HTMLDivElement;

let selectedFile: File | null = null;

// Event Listeners for Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

uploadArea.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files.length > 0) {
    handleFile(fileInput.files[0]);
  }
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  await analyzeImage(selectedFile);
});

function handleFile(file: File) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload a valid image file (JPG, PNG, WEBP).');
    return;
  }
  
  hideError();
  selectedFile = file;
  
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target?.result as string;
    imagePreview.classList.remove('hidden');
    analyzeBtn.disabled = false;
    resultsSection.classList.add('hidden'); // hide previous results
  };
  reader.readAsDataURL(file);
}

function showError(msg: string) {
  errorMessage.textContent = msg;
  errorMessage.classList.remove('hidden');
}

function hideError() {
  errorMessage.classList.add('hidden');
}

function setLoading(isLoading: boolean) {
  if (isLoading) {
    analyzeBtn.disabled = true;
    btnText.textContent = 'Processing...';
    btnLoader.classList.remove('hidden');
  } else {
    analyzeBtn.disabled = false;
    btnText.textContent = 'Analyze Image';
    btnLoader.classList.add('hidden');
  }
}

async function analyzeImage(file: File) {
  setLoading(true);
  hideError();
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}: ${response.statusText}`);
    }
    
    const data: ApiResponse = await response.json();
    displayResults(data);
  } catch (err: any) {
    console.error('API Error:', err);
    showError(`Failed to analyze image. Ensure the backend is running. Error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function displayResults(data: ApiResponse) {
  resultsSection.classList.remove('hidden');
  
  // Animate numbers
  animateValue(resCount, 0, data.predicted_count, 1000);
  animateValue(resGuards, 0, data.security_guards, 800);
  
  // Set heatmap
  resHeatmap.src = `data:image/png;base64,${data.density_map_image}`;
  
  // Set high density direction
  resHighDir.textContent = data.high_density_direction.join(', ') || 'N/A';
  
  // Render Bar chart
  renderBarChart(data.directional_density);
  
  // Scroll to results smoothly
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function renderBarChart(densities: Record<string, number>) {
  barchart.innerHTML = '';
  const entries = Object.entries(densities);
  const maxVal = Math.max(...entries.map(e => e[1])) || 1;
  
  entries.forEach(([dir, val]) => {
    const percentage = Math.max(5, (val / maxVal) * 100);
    const valRounded = Math.round(val * 10) / 10;
    
    const col = document.createElement('div');
    col.className = 'bar-col';
    
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.setAttribute('data-val', valRounded.toString());
    
    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = dir;
    
    col.appendChild(bar);
    col.appendChild(label);
    barchart.appendChild(col);
    
    // Animate height after small delay
    setTimeout(() => {
      bar.style.height = `${percentage}%`;
    }, 50);
  });
}

function animateValue(obj: HTMLElement, start: number, end: number, duration: number) {
  let startTimestamp: number | null = null;
  const step = (timestamp: number) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    
    // easeOutExpo
    const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
    
    obj.innerHTML = Math.floor(easeProgress * (end - start) + start).toString();
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}
