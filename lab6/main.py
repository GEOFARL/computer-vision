import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage import measure, morphology
from scipy.spatial.distance import euclidean
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        logging.error(f"Image not found at path: {path}")
    else:
        logging.info(f"Image loaded from {path}")
    return image

def normalize_image(image, reference):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    image_hsv[..., 2] = cv2.equalizeHist(image_hsv[..., 2])
    reference_hsv[..., 2] = cv2.equalizeHist(reference_hsv[..., 2])
    normalized_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    normalized_reference = cv2.cvtColor(reference_hsv, cv2.COLOR_HSV2BGR)
    logging.info("Images normalized to similar brightness and contrast")
    return normalized_image, normalized_reference

def preprocess_image(image, title="Preprocessed Image"):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv_image, (5, 5), 0)
    logging.info("Image preprocessed for segmentation")
    cv2.imshow(title, blurred)
    cv2.waitKey(0)
    return blurred

def segment_image(image, title="Segmentation Output"):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 200, 200])
    upper_red1 = np.array([40, 255, 255])
    lower_red2 = np.array([130, 200, 200])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    cleaned_mask = morphology.opening(mask, morphology.square(5))
    cleaned_mask = cleaned_mask.astype(np.uint8)
    logging.info("Segmentation based on reddish colors completed")
    cv2.imshow(title, cleaned_mask)
    cv2.imwrite("debug_mask.png", cleaned_mask)
    cv2.waitKey(0)
    return cleaned_mask

def extract_features(mask, min_area=800):
    labeled_image = measure.label(mask)
    features = []
    for region in measure.regionprops(labeled_image):
        if region.area > min_area:
            features.append({
                "label": region.label,
                "area": region.area,
                "centroid": region.centroid,
                "bbox": region.bbox,
                "perimeter": region.perimeter
            })
    logging.info(f"Total labeled regions with area > {min_area}: {len(features)}")
    return features

def cluster_features(features):
    if len(features) == 0:
        logging.warning("No features to cluster. Returning empty labels.")
        return np.array([])
    feature_matrix = np.array([[f["area"]] for f in features])
    n_clusters = min(max(2, len(feature_matrix) // 10), len(feature_matrix))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)
    logging.info("Clustering completed")
    return labels

def classify_object(feature):
    minr, minc, maxr, maxc = feature["bbox"]
    width = maxc - minc
    height = maxr - minr
    aspect_ratio = width / height if height != 0 else 0
    roundness = (4 * np.pi * feature["area"]) / (feature["perimeter"] ** 2) if feature["perimeter"] > 0 else 0
    compactness = feature["area"] / (width * height) if width > 0 and height > 0 else 0
    if feature["area"] > 8000 and compactness < 0.5 and roundness < 0.4:
        return "Quarry or Mining Area"
    elif feature["area"] > 2000 and 0.5 <= aspect_ratio <= 4.0 and compactness > 0.6:
        return "Building or Structure"
    elif feature["area"] > 4000 and roundness > 0.6 and compactness > 0.7:
        return "Lake or Waterbody"
    elif feature["area"] > 1500 and roundness < 0.5 and compactness > 0.5:
        return "Forest or Dense Vegetation"
    elif feature["area"] > 1000 and aspect_ratio > 2.5 and compactness < 0.4:
        return "Road or Path"
    elif feature["area"] > 1200 and compactness > 0.5 and aspect_ratio < 1.5:
        return "Parking Lot or Warehouse"
    elif feature["area"] > 2000 and compactness > 0.4 and roundness < 0.3:
        return "Bare Land"
    else:
        return "Unclassified"

def identify_objects(image, mask, labels, features, title="Identified Objects"):
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 or image.shape[2] == 1 else image.copy()
    color_map = {
        "Quarry or Mining Area": (0, 255, 255),
        "Lake or Waterbody": (0, 255, 0),
        "Building or Structure": (255, 0, 0),
        "Forest or Dense Vegetation": (255, 255, 0),
        "Road or Path": (255, 255, 255),
        "Parking Lot or Warehouse": (128, 0, 128),
        "Bare Land": (128, 128, 0)
    }
    for feature, label in zip(features, labels):
        classification = classify_object(feature)
        if classification == "Unclassified":
            continue
        minr, minc, maxr, maxc = feature["bbox"]
        color = color_map.get(classification, (255, 255, 255))
        cv2.rectangle(output_image, (minc, minr), (maxc, maxr), color, 2)
        label_position = (minc, minr - 10)
        label_text = f"{classification} (Cluster {label})"
        cv2.putText(output_image, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    logging.info("Objects identified and labeled with classifications")
    cv2.imshow(title, output_image)
    cv2.imwrite("filtered_output_refined.png", output_image)
    cv2.waitKey(0)
    return output_image

def compare_objects(features1, features2, area_tolerance=0.6, centroid_distance_tolerance=250, min_iou=0.1):
    comparison_results = []
    for i, feat1 in enumerate(features1):
        best_match, best_iou, best_centroid_dist, best_area_diff = None, 0, float("inf"), float("inf")
        for j, feat2 in enumerate(features2):
            iou = calculate_iou(feat1["bbox"], feat2["bbox"])
            centroid_dist = euclidean(feat1["centroid"], feat2["centroid"])
            area_diff = abs(feat1["area"] - feat2["area"]) / feat1["area"]
            if iou > min_iou and area_diff < area_tolerance and centroid_dist < centroid_distance_tolerance:
                if iou > best_iou:
                    best_iou, best_centroid_dist, best_area_diff, best_match = iou, centroid_dist, area_diff, j
        if best_match is not None:
            comparison_results.append({
                "object_id": i,
                "best_match_id": best_match,
                "iou": best_iou,
                "centroid_distance": best_centroid_dist,
                "area_difference": best_area_diff
            })
    logging.info(f"Matched {len(comparison_results)} objects between images")
    return comparison_results

def calculate_iou(bbox1, bbox2):
    min_r1, min_c1, max_r1, max_c1 = bbox1
    min_r2, min_c2, max_r2, max_c2 = bbox2
    inter_min_r = max(min_r1, min_r2)
    inter_min_c = max(min_c1, min_c2)
    inter_max_r = min(max_r1, max_r2)
    inter_max_c = min(max_c1, max_c2)
    if inter_min_r >= inter_max_r or inter_min_c >= inter_max_c:
        return 0.0
    intersection_area = (inter_max_r - inter_min_r) * (inter_max_c - inter_min_c)
    area1 = (max_r1 - min_r1) * (max_c1 - min_c1)
    area2 = (max_r2 - min_r2) * (max_c2 - min_c2)
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area

def generate_conclusion(comparison_results):
    if not comparison_results:
        logging.info("No matching objects found for comparison.")
        return
    avg_iou = np.mean([result['iou'] for result in comparison_results])
    avg_centroid_distance = np.mean([result['centroid_distance'] for result in comparison_results])
    avg_area_difference = np.mean([result['area_difference'] for result in comparison_results])
    conclusion = (
        f"\n--- Comparison Summary ---\n"
        f"Average IoU: {avg_iou:.2f}\n"
        f"Average Centroid Distance: {avg_centroid_distance:.2f} pixels\n"
        f"Average Area Difference: {avg_area_difference:.2%}\n"
    )
    if avg_iou > 0.4 and avg_area_difference < 0.5:
        conclusion += "The images have a strong similarity in object locations and sizes.\n"
    elif avg_iou > 0.25:
        conclusion += "The images share some similarities, but there are moderate differences.\n"
    else:
        conclusion += "The images show significant differences in object locations and sizes.\n"
    logging.info(conclusion)

def extract_sift_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(contrastThreshold=0.05, edgeThreshold=5)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    logging.info(f"Extracted {len(keypoints)} keypoints from the image.")
    return keypoints, descriptors

def match_sift_descriptors(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.985 * n.distance]
    logging.info(f"Found {len(good_matches)} good matches with ratio {ratio}.")
    return good_matches

def calculate_sift_identification_probability(good_matches, total_keypoints):
    if total_keypoints == 0:
        logging.warning("No keypoints in the source image. Identification probability = 0%.")
        return 0.0
    probability = len(good_matches) / total_keypoints
    logging.info(f"Identification probability: {probability * 100:.2f}%")
    return probability

def draw_matches(image1, keypoints1, image2, keypoints2, good_matches, output_path="matched_output.png"):
    matched_image = cv2.drawMatchesKnn(
        image1, keypoints1, image2, keypoints2, [[m] for m in good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(output_path, matched_image)
    logging.info(f"Matched image saved at {output_path}")
    cv2.imshow("Matched Features", matched_image)
    cv2.waitKey(0)


def main():
    image_high_precision = load_image('high_precision_image.png')
    cv2.imshow("Original High Precision Image", image_high_precision)
    image_operational = load_image('operational_image.png')
    cv2.imshow("Original Operational Image", image_operational)
    cv2.waitKey(0)

    if image_high_precision is not None and image_operational is not None:
        normalized_operational, normalized_high_precision = normalize_image(image_operational, image_high_precision)
        cv2.imshow("Normalized Operational Image", normalized_operational)
        cv2.imshow("Normalized High Precision Image", normalized_high_precision)
        cv2.waitKey(0)

        processed_operational = preprocess_image(normalized_operational, "Preprocessed Operational Image")
        processed_high_precision = preprocess_image(normalized_high_precision, "Preprocessed High Precision Image")

        # Segmentation preview
        segmented_operational = segment_image(processed_operational, "Segmentation Output Operational")
        cv2.imshow("Segmented Operational Image", segmented_operational)
        segmented_high_precision = segment_image(processed_high_precision, "Segmentation Output High Precision")
        cv2.imshow("Segmented High Precision Image", segmented_high_precision)
        cv2.waitKey(0)

        features_operational = extract_features(segmented_operational, min_area=1000)
        features_high_precision = extract_features(segmented_high_precision, min_area=1000)
        logging.info(f"Operational Features: {features_operational}")
        logging.info(f"High Precision Features: {features_high_precision}")

        labels_operational = cluster_features(features_operational)
        labels_high_precision = cluster_features(features_high_precision)
        logging.info(f"Operational Clusters: {labels_operational}")
        logging.info(f"High Precision Clusters: {labels_high_precision}")

        identified_operational = identify_objects(processed_operational, segmented_operational, labels_operational, features_operational, "Identified Objects Operational")
        identified_high_precision = identify_objects(processed_high_precision, segmented_high_precision, labels_high_precision, features_high_precision, "Identified Objects High Precision")

        # Comparison preview
        comparison_results = compare_objects(features_operational, features_high_precision, area_tolerance=0.4, centroid_distance_tolerance=200)
        logging.info(f"Comparison Results: {comparison_results}")
        generate_conclusion(comparison_results)
        cv2.waitKey(0)

    keypoints_operational, descriptors_operational = extract_sift_features(processed_operational)
    keypoints_high_precision, descriptors_high_precision = extract_sift_features(processed_high_precision)

    if descriptors_operational is None or descriptors_high_precision is None:
        logging.error("SIFT descriptors could not be extracted from one or both images.")
        return


    keypoints1, descriptors1 = extract_sift_features(image_high_precision)
    keypoints2, descriptors2 = extract_sift_features(image_operational)
    good_matches = match_sift_descriptors(descriptors1, descriptors2)

    probability = calculate_sift_identification_probability(good_matches, len(keypoints1))

    draw_matches(image_high_precision, keypoints1, image_operational, keypoints2, good_matches)

    print(f"Ймовірність ідентифікації: {probability * 100:.2f}%")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
