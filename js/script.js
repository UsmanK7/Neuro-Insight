document.addEventListener('DOMContentLoaded', function () {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const sampleImages = document.getElementById('sampleImages');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressBar = uploadProgress.querySelector('.progress-bar');
    const successCheckmark = uploadProgress.querySelector('.success-checkmark');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        });
    });

    // Handle file drop
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => handleFiles(fileInput.files));

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                uploadFile(file);
            }
        }
    }

    function uploadFile(file) {
        sampleImages.classList.add('hide');
        uploadProgress.style.display = 'block';
        progressBar.style.width = '0%';
        successCheckmark.style.display = 'none';

        // Simulate file upload
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            progressBar.style.width = progress + '%';
            if (progress >= 100) {
                clearInterval(interval);
                successCheckmark.style.display = 'block';
            }
        }, 100);
    }

    // Handle sample image selection
    document.querySelectorAll('.sample-image').forEach(img => {
        img.addEventListener('click', () => {
            // Handle sample image selection here
            console.log('Selected sample image:', img.src);
        });
    });
});
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz"
crossorigin="anonymous"></script>
<script>
document.getElementById('drawerToggle').addEventListener('click', function (event) {
    event.stopPropagation();
    document.getElementById('drawer').classList.toggle('show');
});

document.addEventListener('click', function (event) {
    var drawer = document.getElementById('drawer');
    if (drawer.classList.contains('show') && !drawer.contains(event.target)) {
        drawer.classList.remove('show');
    }
});