import logging
import os
from typing import Any, Dict, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
import fitz # PyMuPDF
from pathlib import Path

from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.data_objects import DocumentContext, PageContext
from ocrx.core.exceptions import OCRXConfigurationError, OCRXInputError, OCRXProcessingError

class ImageLoader(OCRXModuleBase):
    """
    Loads images from various sources (file paths, bytes, PDFs) into the DocumentContext.
    """
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.default_dpi = 300 # Default DPI for PDF rendering

    def _validate_config(self) -> None:
        super()._validate_config()
        self.default_dpi = self.config.get("default_dpi", 300)
        if not isinstance(self.default_dpi, int) or self.default_dpi <= 0:
            raise OCRXConfigurationError(
                f"{self.module_id} config: 'default_dpi' must be a positive integer. Found: {self.default_dpi}"
            )
        self.logger.info(f"{self.module_id} configured with default_dpi: {self.default_dpi}")

    def process(self, document_context: DocumentContext, source: Union[str, bytes]) -> DocumentContext:
        """
        Loads images from the source and populates the document_context.

        Args:
            document_context: The DocumentContext object to populate.
            source: The image source, can be a file path (str) or image bytes.
                    URLs are a TODO.

        Returns:
            The populated DocumentContext.
        """
        self.logger.info(f"Processing source: {document_context.document_id} (type: {type(source)})")
        document_context.overall_status = "loading_images"

        try:
            if isinstance(source, str): # File path
                if not os.path.exists(source):
                    raise OCRXInputError(f"File not found: {source}")

                file_ext = Path(source).suffix.lower()
                if file_ext == ".pdf":
                    self._load_pdf(document_context, source)
                elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"]:
                    self._load_image_file(document_context, source)
                else:
                    raise OCRXInputError(f"Unsupported file extension: {file_ext} for source: {source}")

            elif isinstance(source, bytes):
                # Try to open as an image first, then as PDF if that fails.
                # This is a basic heuristic. More robust type detection might be needed.
                try:
                    self._load_image_bytes(document_context, source)
                except UnidentifiedImageError:
                    self.logger.debug("Failed to load source bytes as image, attempting PDF.")
                    try:
                        self._load_pdf_bytes(document_context, source)
                    except Exception as pdf_e: # fitz might raise various errors
                        raise OCRXInputError(f"Source bytes could not be processed as a known image type or PDF. PDF error: {pdf_e}") from pdf_e
            else:
                raise OCRXInputError(f"Unsupported source type: {type(source)}. Must be str (path) or bytes.")

            if not document_context.pages:
                raise OCRXProcessingError("No pages were loaded from the source.")

            document_context.overall_status = "images_loaded"
            self.logger.info(f"Successfully loaded {len(document_context.pages)} page(s) for {document_context.document_id}")

        except (OCRXInputError, OCRXProcessingError, FileNotFoundError, UnidentifiedImageError, fitz.FitzError) as e:
            self.logger.error(f"Error during image loading for {document_context.document_id}: {e}", exc_info=True)
            document_context.document_errors.append(f"ImageLoader: {str(e)}")
            document_context.overall_status = "loading_failed"

        return document_context

    def _pil_to_numpy_bgr(self, pil_image: Image.Image) -> np.ndarray:
        """Converts a Pillow image to a NumPy BGR array."""
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB') # Drop alpha for consistency
        elif pil_image.mode == 'L' or pil_image.mode == '1': # Grayscale or Binary
             pil_image = pil_image.convert('RGB') # Convert to RGB
        elif pil_image.mode != 'RGB':
            # Attempt conversion, but log if unusual mode
            self.logger.warning(f"Image in unusual mode '{pil_image.mode}', attempting RGB conversion.")
            pil_image = pil_image.convert('RGB')

        # Convert RGB Pillow image to BGR NumPy array (OpenCV standard)
        return np.array(pil_image)[:, :, ::-1]


    def _load_image_file(self, document_context: DocumentContext, image_path: str) -> None:
        """Loads a single image file (PNG, JPG, TIFF, BMP) into a PageContext."""
        self.logger.debug(f"Loading image file: {image_path}")
        try:
            pil_img = Image.open(image_path)
            np_bgr_image = self._pil_to_numpy_bgr(pil_img)

            page_ctx = PageContext(
                page_number=len(document_context.pages),
                original_image=np_bgr_image
            )
            document_context.pages.append(page_ctx)
        except UnidentifiedImageError as e:
            raise OCRXInputError(f"Cannot identify image file (unsupported format or corrupted): {image_path}") from e
        except Exception as e:
            raise OCRXProcessingError(f"Unexpected error loading image file {image_path}: {e}") from e


    def _load_image_bytes(self, document_context: DocumentContext, image_bytes: bytes) -> None:
        """Loads a single image from bytes into a PageContext."""
        self.logger.debug("Loading image from bytes.")
        import io
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            np_bgr_image = self._pil_to_numpy_bgr(pil_img)

            page_ctx = PageContext(
                page_number=len(document_context.pages),
                original_image=np_bgr_image
            )
            document_context.pages.append(page_ctx)
        except UnidentifiedImageError as e:
            # Re-raise specifically so the caller can try PDF if this was the first attempt
            raise UnidentifiedImageError("Bytes could not be identified as a standard image format.") from e
        except Exception as e:
            raise OCRXProcessingError(f"Unexpected error loading image from bytes: {e}") from e


    def _load_pdf(self, document_context: DocumentContext, pdf_path: str) -> None:
        """Loads pages from a PDF file into multiple PageContexts."""
        self.logger.debug(f"Loading PDF file: {pdf_path} with DPI: {self.default_dpi}")
        try:
            pdf_doc = fitz.open(pdf_path)
            for i, fitz_page in enumerate(pdf_doc):
                pix = fitz_page.get_pixmap(dpi=self.default_dpi)
                if pix.alpha: # Pixmap has alpha channel
                    # Create an RGBA image then convert to RGB (dropping alpha)
                    np_image_rgba = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)
                    np_image_rgb = np_image_rgba[:, :, :3]
                else: # Pixmap is RGB
                    np_image_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

                # Convert RGB to BGR (OpenCV standard)
                np_bgr_image = np_image_rgb[:, :, ::-1].copy() # Ensure it's a writable copy

                page_ctx = PageContext(
                    page_number=i, # 0-indexed
                    original_image=np_bgr_image
                )
                document_context.pages.append(page_ctx)
                self.logger.debug(f"Loaded page {i} from PDF {pdf_path}, dimensions: {np_bgr_image.shape}")
            pdf_doc.close()
        except fitz.FitzError as e:
            raise OCRXInputError(f"PyMuPDF (Fitz) error processing PDF {pdf_path}: {e}") from e
        except Exception as e:
            raise OCRXProcessingError(f"Unexpected error loading PDF {pdf_path}: {e}") from e

    def _load_pdf_bytes(self, document_context: DocumentContext, pdf_bytes: bytes) -> None:
        """Loads pages from PDF bytes into multiple PageContexts."""
        self.logger.debug(f"Loading PDF from bytes with DPI: {self.default_dpi}")
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for i, fitz_page in enumerate(pdf_doc):
                pix = fitz_page.get_pixmap(dpi=self.default_dpi)
                if pix.alpha:
                    np_image_rgba = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)
                    np_image_rgb = np_image_rgba[:, :, :3]
                else:
                    np_image_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)

                np_bgr_image = np_image_rgb[:, :, ::-1].copy()

                page_ctx = PageContext(
                    page_number=i,
                    original_image=np_bgr_image
                )
                document_context.pages.append(page_ctx)
                self.logger.debug(f"Loaded page {i} from PDF bytes, dimensions: {np_bgr_image.shape}")
            pdf_doc.close()
        except fitz.FitzError as e:
            raise OCRXInputError(f"PyMuPDF (Fitz) error processing PDF from bytes: {e}") from e
        except Exception as e:
            raise OCRXProcessingError(f"Unexpected error loading PDF from bytes: {e}") from e
