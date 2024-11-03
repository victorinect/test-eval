import os
import cv2
import numpy as np
import pandas as pd

class StyleEvaluationAgent:
    def __init__(self, style_dataset):
        self.dataset = style_dataset
        self.results = []

    def analyze_image(self, image_path):
        print("image path", image_path)
        image = cv2.imread(image_path)
    
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return
        
        # Design Element Analysis
        element_attributes = self.design_element_analysis(image)

        # Design Principle Analysis
        principle_attributes = self.design_principle_analysis(image)

        # Feature Extraction
        specific_image_attributes = self.extract_features(image)

        # Combine attributes into a single dictionary for this image
        combined_attributes = {
            **element_attributes,
            **principle_attributes,
            **specific_image_attributes
        }
        
        self.results.append(combined_attributes)

    def design_element_analysis(self, image):
        attributes = {
            'line_density': self.calculate_line_density(image),
            'shape_complexity': self.calculate_shape_complexity(image),
            'colorfulness': self.calculate_colorfulness(image),
            'texture_roughness': self.calculate_texture_roughness(image),
            'brightness': self.calculate_brightness(image)
        }
        return attributes

    def design_principle_analysis(self, image):
        attributes = {
            'contrast': self.calculate_contrast(image),
            'symmetry': self.calculate_symmetry(image),
            'warmth_ratio': self.calculate_warmth_ratio(image),
            'proportion_of_space': self.calculate_proportion_of_space(image)
        }
        return attributes

    def extract_features(self, image):
        attributes = {
            'edge_density': self.calculate_edge_density(image)
        }
        return attributes

    def calculate_line_density(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        line_density = np.sum(edges > 0) / edges.size
        return line_density

    def calculate_shape_complexity(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_complexity = len(contours)
        return shape_complexity

    def calculate_colorfulness(self, image):
        (B, G, R) = cv2.split(image.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
        mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
        colorfulness = std_root + (0.3 * mean_root)
        return colorfulness

    def calculate_texture_roughness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_roughness = np.std(gray)
        return texture_roughness

    def calculate_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        return brightness

    def calculate_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        return contrast

    def calculate_edge_density(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density

    def calculate_symmetry(self, image):
        half_width = image.shape[1] // 2
        left_half = image[:, :half_width]
        right_half = image[:, half_width:]

        # If the width is odd, right_half will have one extra column
        if left_half.shape[1] != right_half.shape[1]:
            right_half = right_half[:, :half_width]  # Trim right_half to match left_half

        right_half_flipped = cv2.flip(right_half, 1)
        symmetry = np.mean(left_half == right_half_flipped)
        return symmetry


    def calculate_warmth_ratio(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        warm_pixels = np.sum((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150))
        total_pixels = hsv.shape[0] * hsv.shape[1]
        warmth_ratio = warm_pixels / total_pixels
        return warmth_ratio

    def calculate_proportion_of_space(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        background_ratio = np.sum(thresh == 255) / thresh.size
        return background_ratio

    def analyze_dataset(self):
        # Analyze each image in the dataset
        for image_path in self.dataset:
            self.analyze_image(image_path)

    def export_to_csv(self, output_path):
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    theme = "underwater"
    dataset = [os.path.join(f"./{theme}", f) for f in os.listdir(f"./{theme}") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    reference_img = f"./reference-imgs/pass.jpg"
    ds_agent = StyleEvaluationAgent(dataset)
    reference_agent = StyleEvaluationAgent([reference_img])

    # Perform analysis
    ds_agent.analyze_dataset()
    reference_agent.analyze_dataset()

    # Export results to a CSV
    ds_agent.export_to_csv(f"./results/underwater/style_analysis_results.csv")
    reference_agent.export_to_csv(f"./results/underwater/reference-img-analysis_results.csv")

    print(os.listdir("./results/underwater/"))

