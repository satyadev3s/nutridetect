document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('photoUploadContainer');
    const fileInput = document.getElementById('profilePhoto');
    const photoPreview = document.getElementById('photoPreview');
    const photoPreviewContainer = document.getElementById('photoPreviewContainer');

    if (uploadZone && fileInput) {
        uploadZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (event) => {
            const [file] = event.target.files || [];
            if (!file || !photoPreview || !photoPreviewContainer) {
                return;
            }

            const reader = new FileReader();
            reader.onload = (loadEvent) => {
                photoPreview.src = loadEvent.target.result;
                photoPreviewContainer.style.display = 'block';
                uploadZone.style.display = 'none';
            };
            reader.readAsDataURL(file);
        });
    }
});

function clearPhoto() {
    const uploadZone = document.getElementById('photoUploadContainer');
    const fileInput = document.getElementById('profilePhoto');
    const photoPreview = document.getElementById('photoPreview');
    const photoPreviewContainer = document.getElementById('photoPreviewContainer');

    if (fileInput) {
        fileInput.value = '';
    }
    if (photoPreview) {
        photoPreview.src = '';
    }
    if (photoPreviewContainer) {
        photoPreviewContainer.style.display = 'none';
    }
    if (uploadZone) {
        uploadZone.style.display = 'block';
    }
}

function showNutriAlert(icon, title, text) {
    if (typeof Swal === 'undefined') {
        window.alert(`${title}\n\n${text}`);
        return;
    }

    Swal.fire({
        icon,
        title,
        text,
        customClass: {
            popup: 'nutri-alert-popup',
            title: 'nutri-alert-title',
            htmlContainer: 'nutri-alert-text',
            confirmButton: 'nutri-alert-confirm',
        },
        background: '#0f172a',
        color: '#e8f7ff',
        confirmButtonText: 'OK',
        confirmButtonColor: '#00d4ff',
        focusConfirm: true,
        allowEscapeKey: true,
        allowOutsideClick: true,
        heightAuto: false,
    });
}
