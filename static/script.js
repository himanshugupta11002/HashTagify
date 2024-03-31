// static/js/app.js

document.addEventListener("DOMContentLoaded", function () {
    const imagesInput = document.getElementById("images");
    const previewContainer = document.getElementById("image-preview");
    const previewImagesContainer = document.getElementById("preview-container");

    imagesInput.addEventListener("change", function () {
        previewImagesContainer.innerHTML = ""; // Clear previous previews

        for (const file of this.files) {
            const reader = new FileReader();

            reader.addEventListener("load", function () {
                const imgElement = document.createElement("img");
                imgElement.src = reader.result;
                imgElement.alt = "Selected Image";
                imgElement.className = "preview-image";
                previewImagesContainer.appendChild(imgElement);
            });

            reader.readAsDataURL(file);
        }

        previewContainer.style.display = this.files.length > 0 ? "block" : "none";
    });
});
