import cv2
import numpy as np
from PIL import Image
import logging
import torch
import openai
import clip
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

# Import SAM2 modules from the official Segment Anything repository.
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# Import GroundingDINO inference utilities.
from groundingdino.util.inference import load_model, predict

# Configure logging for production-grade feedback.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
load_dotenv()


class ImageTransformer:
    def __init__(self,
                 desired_scale_ratio=0.8,
                 product_description="a modern coffee maker on a kitchen counter",
                 target_region="counter",
                 text_prompt="all objects",
                 box_threshold=0.3,
                 text_threshold=0.25,
                 sam_checkpoint="sam2/sam_vit_h_4b8939.pth",  # Update to your SAM2 checkpoint.
                 sam_model_type="vit_h",
                 grounddino_config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",  # Update path.
                 grounddino_checkpoint="GroundingDINO/groundingdino/groundingdino_swint_ogc.pth",  # Update path.
                 device=None):
        """
        Initialize the transformer.
        
        Args:
          - desired_scale_ratio (float): Fraction of the candidate region’s width that the product should occupy.
          - product_description (str): Description of the product and its typical placement.
          - target_region (str): Keyword (e.g., "counter") to help select the region from the scene graph.
          - text_prompt (str): Text query used with GroundingDINO.
          - box_threshold, text_threshold (float): Detection thresholds for GroundingDINO.
          - sam_checkpoint, sam_model_type: Parameters for SAM2.
          - grounddino_config, grounddino_checkpoint: Parameters for GroundingDINO.
          - device (str): 'cuda' or 'cpu' (if None, uses GPU if available).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.desired_scale_ratio = desired_scale_ratio
        self.product_description = product_description
        self.target_region = target_region.lower()
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Set OpenAI API key. For production, load from an environment variable.
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Load SAM2 model.
        logging.info("Loading SAM2 model...")
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # Load GroundingDINO model.
        logging.info("Loading GroundingDINO model...")
        self.grounddino_model = load_model(grounddino_config, grounddino_checkpoint, device=device)

        # Load CLIP model for semantic matching.
        logging.info("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.to(device)
        self.product_text_embedding = self.get_text_embedding(self.product_description)

    def get_text_embedding(self, text):
        """
        Returns a normalized text embedding for the given text using CLIP.
        """
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding

    def run_sam2_segmentation(self, image_np):
        """
        Runs SAM2 segmentation on the provided RGB image (numpy array).
        Returns a list of binary masks.
        """
        logging.info("Running SAM2 segmentation...")
        masks = self.mask_generator.generate(image_np)
        binary_masks = [mask_dict["segmentation"].astype(np.uint8) for mask_dict in masks]
        logging.info(f"SAM2 generated {len(binary_masks)} mask(s).")
        return binary_masks

    def run_groundingdino_detection(self, image_np):
        """
        Runs GroundingDINO detection on the provided RGB image (numpy array).
        Returns bounding boxes and phrases.
        """
        logging.info("Running GroundingDINO detection...")
        boxes, scores, phrases = predict(
            self.grounddino_model,
            image_np,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold
        )
        logging.info(f"GroundingDINO found {len(boxes)} box(es).")
        return boxes, phrases

    def generate_scene_graph(self, image_np):
        """
        Generates a scene graph by using GroundingDINO's detection.
        Returns a list of objects with labels and bounding boxes.
        """
        boxes, phrases = self.run_groundingdino_detection(image_np)
        scene_graph = []
        for box, phrase in zip(boxes, phrases):
            x1, y1, x2, y2 = map(int, box)
            scene_graph.append({
                "label": phrase,
                "bbox": [x1, y1, x2, y2]
            })
        logging.info(f"Generated scene graph with {len(scene_graph)} objects.")
        return scene_graph

    def optimize_scene_graph_with_openai(self, scene_graph):
        """
        Uses OpenAI's ChatCompletion API to optimize the scene graph.
        The prompt asks which object (by number) is best for product placement.
        Returns the chosen candidate's bounding box (and a default orientation).
        """
        # Construct a text summary of the scene graph.
        scene_graph_text = ""
        for i, obj in enumerate(scene_graph):
            bbox = obj["bbox"]
            label = obj["label"]
            scene_graph_text += f"Object {i+1}: Label: {label}, BBox: {bbox}\n"
        
        prompt = (
            "You are an expert interior designer. Given the following scene graph from an image:\n"
            f"{scene_graph_text}\n"
            f"And given the product description: '{self.product_description}', "
            "please identify which object (by number) represents the best region for placing the product. "
            "Respond with only the object number."
        )
        
        logging.info("Sending prompt to OpenAI for scene graph optimization...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            chosen_object_str = response["choices"][0]["message"]["content"].strip()
            logging.info(f"OpenAI response: {chosen_object_str}")
            chosen_index = int(chosen_object_str) - 1  # Adjust for 0-based indexing.
        except Exception as e:
            logging.error(f"Error with OpenAI API: {e}")
            chosen_index = None
        
        if chosen_index is not None and 0 <= chosen_index < len(scene_graph):
            chosen_obj = scene_graph[chosen_index]
            logging.info(f"Optimized scene graph chose object {chosen_index+1} with label '{chosen_obj['label']}' and bbox {chosen_obj['bbox']}")
            return chosen_obj["bbox"], 0.0  # We return a default angle of 0°.
        else:
            logging.warning("OpenAI optimization failed; no candidate selected.")
            return None, 0.0

    def compute_safe_zone_and_angle(self, background_image):
        """
        Computes the safe zone by first generating a scene graph from the background,
        then optimizing it via OpenAI. If this fails, falls back to using SAM2 segmentation.
        Returns a bounding box (x, y, w, h) and an orientation angle.
        """
        bg_np = np.array(background_image.convert("RGB"))
        scene_graph = self.generate_scene_graph(bg_np)
        bbox, angle = self.optimize_scene_graph_with_openai(scene_graph)
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_converted = (x1, y1, x2 - x1, y2 - y1)
            return bbox_converted, angle
        else:
            logging.info("Falling back to SAM2 segmentation for safe zone estimation...")
            binary_masks = self.run_sam2_segmentation(bg_np)
            if binary_masks:
                combined_mask = np.clip(np.sum(binary_masks, axis=0), 0, 1).astype(np.uint8)
            else:
                h, w, _ = bg_np.shape
                return (0, 0, w, h), 0.0
            safe_mask = 1 - combined_mask
            safe_mask_uint8 = (safe_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(safe_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                h, w, _ = bg_np.shape
                return (0, 0, w, h), 0.0
            best_contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(best_contour)
            rect = cv2.minAreaRect(best_contour)
            angle = rect[2]
            if angle < -45:
                angle += 90
            return bbox, angle

    def calculate_transform_params(self, background_image, product_image):
        """
        Calculates scaling, rotation, and translation parameters for the product image.
        Returns the scale factor, rotation angle, and top-left placement coordinates.
        """
        bbox, angle = self.compute_safe_zone_and_angle(background_image)
        x, y, bw, bh = bbox
        new_prod_width = bw * self.desired_scale_ratio
        scale_factor = new_prod_width / product_image.width
        new_prod_height = product_image.height * scale_factor
        center_x, center_y = x + bw / 2, y + bh / 2
        x_pos = center_x - new_prod_width / 2
        y_pos = center_y - new_prod_height / 2
        logging.info(f"Calculated scale: {scale_factor:.2f}, position: ({x_pos:.2f}, {y_pos:.2f}), rotation: {angle:.2f}°")
        return scale_factor, angle, (x_pos, y_pos)

    def transform_product_image(self, product_image, scale_factor, angle):
        """
        Rotates and scales the product image accordingly.
        Returns the transformed product image.
        """
        product_cv = cv2.cvtColor(np.array(product_image), cv2.COLOR_RGBA2BGRA)
        h, w = product_cv.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale_factor)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated_scaled = cv2.warpAffine(
            product_cv, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        product_transformed = Image.fromarray(cv2.cvtColor(rotated_scaled, cv2.COLOR_BGRA2RGBA))
        return product_transformed

    def composite_images(self, background_image, product_transformed, position):
        """
        Overlays the transformed product image onto the background.
        Returns the final composite image.
        """
        bg_copy = background_image.copy()
        bg_copy.paste(product_transformed, (int(position[0]), int(position[1])), product_transformed)
        return bg_copy


def process_images(background, product, transformer):
    scale, angle, position = transformer.calculate_transform_params(background, product)
    product_transformed = transformer.transform_product_image(product, scale, angle)
    final_image = transformer.composite_images(background, product_transformed, position)
    return final_image

def load_image_from_url(url):
    """
    Load an image from a URL and convert it to PIL Image.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    return Image.open(BytesIO(response.content))

def main():
    try:
        background_url = "https://media.istockphoto.com/id/1428205261/photo/modern-and-luxury-beige-wooden-kitchen-island-and-counter-with-white-counter-top-and-glass.jpg?s=1024x1024&w=is&k=20&c=FsuVhUadeD-d25JHBUotOsm15n7LXraejiSAsqvs0pE="
        product_url = "https://media.istockphoto.com/id/1406569643/vector/drip-coffee-maker-3d-vector-illustration-realistic-coffee-machine-isolated-on-transparent.jpg?s=1024x1024&w=is&k=20&c=WKAx_RiL9FxztS1SMpKA9KJVGzDBhjwJ3dnctab63D8="
        background = load_image_from_url(background_url).convert("RGBA")
        product = load_image_from_url(product_url).convert("RGBA")
    except Exception as e:
        logging.error(f"Error loading images: {e}")
        return

    # Update the following paths as needed.
    sam_checkpoint = "sam2/sam_vit_h_4b8939.pth"
    grounddino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounddino_checkpoint = "GroundingDINO/groundingdino/groundingdino_swint_ogc.pth"

    transformer = ImageTransformer(
        desired_scale_ratio=0.8,
        product_description="a modern coffee maker on a kitchen counter",
        target_region="counter",
        text_prompt="all objects",
        box_threshold=0.3,
        text_threshold=0.25,
        sam_checkpoint=sam_checkpoint,
        sam_model_type="vit_h",
        grounddino_config=grounddino_config,
        grounddino_checkpoint=grounddino_checkpoint
    )
    final_image = process_images(background, product, transformer)
    try:
        final_image.save("final_composite.png")
        logging.info("Final composite image saved as 'final_composite.png'.")
    except Exception as e:
        logging.error(f"Error saving final image: {e}")


if __name__ == "__main__":
    main()