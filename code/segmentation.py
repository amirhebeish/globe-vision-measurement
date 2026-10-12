import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def watershed(src_img, blue_threshold, plot_process):

    # Blue dominance preprocessing
    b, g, r = cv2.split(src_img)
    blue_score = b.astype(np.int16) - ((r.astype(np.int16) + g.astype(np.int16)) // 2)
    blue_score = np.clip(blue_score + 128, 0, 255).astype(np.uint8)
    blue_blur = cv2.GaussianBlur(blue_score, (7, 7), 0)

    # Threshold to keep strong blue pixels as white
    T_blue = blue_threshold 
    _, blue_bin = cv2.threshold(blue_blur, T_blue, 255, cv2.THRESH_BINARY)

    # Decide whether fragments should be glued or split
    contours, _ = cv2.findContours(blue_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
    should_glue = False # Default to NO glue (safer for cluttered images)
    if len(significant_contours) > 0:
        all_points = np.vstack(significant_contours)
        
        # Solidity measures how compact the group of regions is
        hull = cv2.convexHull(all_points)
        hull_area = cv2.contourArea(hull)
        total_blue_area = sum(cv2.contourArea(c) for c in significant_contours)
        solidity = 0
        if hull_area > 0:
            solidity = total_blue_area / float(hull_area)
            
        # Centrality ensures the object is not side clutter
        x, y, w, h = cv2.boundingRect(all_points)
        group_center_x = x + w/2
        image_center_x = blue_bin.shape[1] / 2
        dist_from_center = abs(group_center_x - image_center_x)

        is_dense = solidity > 0.5
        is_central = dist_from_center < (blue_bin.shape[1] / 3.5)
        
        if is_dense and is_central:
            should_glue = True

    # Apply morphology based on the decision
    if should_glue:
        kernel_glue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        blue_clean = cv2.morphologyEx(blue_bin, cv2.MORPH_CLOSE, kernel_glue)
    else:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blue_clean = cv2.morphologyEx(blue_bin, cv2.MORPH_OPEN, kernel_open)

    
    
    if plot_process:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].imshow(src_img);          ax[0].set_title('Original');            ax[0].axis('off')
        ax[1].imshow(blue_score, cmap='gray'); ax[1].set_title('Blue-dominance');   ax[1].axis('off')
        ax[2].imshow(blue_blur,  cmap='gray'); ax[2].set_title('Blue-dominance (blurred)'); ax[2].axis('off')
        ax[3].imshow(blue_bin, cmap='gray'); ax[3].set_title(f'Binary mask (T = {T_blue})')
        plt.tight_layout()
        plt.show()

    # Connected components: select largest blue region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_clean)
    if num_labels <= 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_bin)

    if num_labels <= 1:
        empty_mask = np.zeros(src_img.shape[:2], dtype=np.uint8)
        return src_img, empty_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    #  Create markers for watershed segmentation
    markers = np.ones_like(labels, dtype=np.int32)
    markers[labels == largest_label] = 2

    if plot_process:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].imshow(blue_bin, cmap='gray'); ax[0].set_title('Blue-based binary mask')
        ax[1].imshow(markers, cmap='jet');   ax[1].set_title('Markers (1 = bg, 2 = globe)'); ax[1].axis('off')
        plt.tight_layout()
        plt.show()

    # Watershed refinement
    markers_ws = cv2.watershed(src_img, markers)
    ao_mask = np.uint8(markers_ws == 2) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ao_mask = cv2.morphologyEx(ao_mask, cv2.MORPH_CLOSE, kernel_small)

    # Smooth final mask using circle or ellipse fitting 
    contours, _ = cv2.findContours(ao_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_type = None   

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)

        if len(cnt) >= 5:

            hull = cv2.convexHull(cnt)
        
            if len(hull) > 4: ellipse = cv2.fitEllipse(hull)  
            else: ellipse = cv2.fitEllipse(cnt)   
            
            (cx, cy), (MA, ma), angle = ellipse
            axis_ratio = max(MA, ma) / min(MA, ma)

            ao_mask_smooth = np.zeros_like(ao_mask)

            if axis_ratio < 1.05:

                radius = int((MA + ma) / 4.0)        
                center = (int(cx), int(cy))
                cv2.circle(ao_mask_smooth, center, radius, 255, -1)
                shape_type = "circle"
                circle_center, circle_radius = center, radius
            else:
                cv2.ellipse(ao_mask_smooth, ellipse, 255, -1)
                shape_type = "ellipse"

            ao_mask = ao_mask_smooth

    # Overlay the fitted boundary on the original image
    overlay_bgr = src_img

    if shape_type == "circle":
        cv2.circle(overlay_bgr, circle_center, circle_radius, (0, 0, 255), 3)
    elif shape_type == "ellipse":
        cv2.ellipse(overlay_bgr, ellipse, (0, 0, 255), 3)

    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay, ao_mask

def preprocessing(img):

    # Normalize illumination and reduce shadows using LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab) 

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab_eq = cv2.merge((l, a, b))

    lab_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2RGB)

    # Smooth to suppress internal edges
    processed_img = cv2.GaussianBlur(lab_eq, (9, 9), 0) # avoid strong edges inside AO

    return processed_img

def get_kmeans(preprocessed_img, n_clusters):

    # Apply K-means clustering in RGB space
    img_flat = preprocessed_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_flat)

    H, W, _ = preprocessed_img.shape
    labels_img = kmeans.labels_.reshape(H, W)

    return labels_img

def remove_bg(labels_img, rgb):

    # Select the cluster most likely corresponding to the globe
    h, w = labels_img.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    center_y_min, center_y_max = int(h * 0.25), int(h * 0.75)
    center_x_min, center_x_max = int(w * 0.25), int(w * 0.75)
    
    unique_labels = np.unique(labels_img)

    biggest_cluster_ratio = -1
    main_cluster = -1

    for i in unique_labels:
        mask = (labels_img == i).astype(np.uint8) * 255

        rgb_masked = cv2.bitwise_and(rgb, rgb, mask=mask)

        total_pixels = np.count_nonzero(mask)

        red_avg = np.sum(rgb_masked[:, :, 0]) / total_pixels
        
        if total_pixels == 0 or red_avg > 100: continue
        
        central_region = mask[center_y_min:center_y_max, center_x_min:center_x_max]
        
        central_pixels = np.count_nonzero(central_region)
        
        density_ratio = central_pixels / total_pixels

        if density_ratio > biggest_cluster_ratio:

            biggest_cluster_ratio = density_ratio
            main_cluster = i
    
    final_mask = (labels_img == main_cluster).astype(np.uint8) * 255

    return final_mask

def show_process(src_img, img_pre, labels_img, closed_mask, ellipse_mask, result_kmeans):

    # Visualise main steps of the K-means pipeline
    fig, axis = plt.subplots(2, 3, figsize=(15, 6))
    axis[0][0].imshow(src_img)
    axis[0][0].set_title('Original')

    axis[0][1].imshow(img_pre)
    axis[0][1].set_title('Preprocessed')

    axis[0][2].imshow(labels_img)
    axis[0][2].set_title('Segmented')

    axis[1][0].imshow(closed_mask, 'gray')
    axis[1][0].set_title('closed mask')

    axis[1][1].imshow(ellipse_mask, 'gray')
    axis[1][1].set_title('ellipse mask')

    axis[1][2].imshow(result_kmeans)
    axis[1][2].set_title('result')

    return None

def kmeans_segmentation(src_img, threshold, clusters, plot):

    # Full K-means based segmentation pipeline
    img_pre = preprocessing(src_img)
    n_clusters = clusters
    labels_img = get_kmeans(img_pre, n_clusters)
    kmeans_mask = remove_bg(labels_img, src_img.copy())

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(kmeans_mask, cv2.MORPH_CLOSE, kernel)

    # Remove outliers based on distance from center of mass
    closed_mask_points = cv2.findNonZero(closed_mask)
    hull_mask = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
    points_array = closed_mask_points.reshape(-1, 2)

    center_of_mass = np.mean(points_array, axis=0)
    distances = np.linalg.norm(points_array - center_of_mass, axis=1)

    threshold_distance = np.percentile(distances, threshold)
    points_to_keep_mask = distances <= threshold_distance
    kept_points = points_array[points_to_keep_mask]

    # Remove outliers based on distance from center of mass
    convex_hull = cv2.convexHull(kept_points)
    hull_mask = cv2.drawContours(hull_mask, [convex_hull], -1, 255, thickness=cv2.FILLED)

    if len(convex_hull) > 4: ellipse_kmeans = cv2.fitEllipse(convex_hull)
    else: ellipse_kmeans = cv2.fitEllipse(kept_points)

    ellipse_mask = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
    ellipse_mask = cv2.ellipse(ellipse_mask, ellipse_kmeans, 255, -1)

    result_kmeans = cv2.ellipse(src_img, ellipse_kmeans, (255, 0, 0), 2)

    if plot:

        show_process(src_img, img_pre, labels_img, closed_mask, ellipse_mask, result_kmeans)

    return result_kmeans, ellipse_mask

def segment_globe_regiongrow_shape_param(src_rgb, white_v_min=170, plot_process=False):
   
    H, W = src_rgb.shape[:2]

    # Preprocess 
    pre = preprocess_for_globe(src_rgb)

    # Colour prior for globe (blue and white)
    hsv = cv2.cvtColor(pre, cv2.COLOR_RGB2HSV)
 
    lower_blue  = np.array([90, 40, 40], np.uint8)
    upper_blue  = np.array([150, 255, 255], np.uint8)
    blue_mask   = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([0, 0, white_v_min], np.uint8)
    upper_white = np.array([179, 80, 255], np.uint8)
    white_mask  = cv2.inRange(hsv, lower_white, upper_white)

    globe_prior = cv2.bitwise_or(blue_mask, white_mask)

    # Morphological to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    globe_prior = cv2.morphologyEx(globe_prior, cv2.MORPH_CLOSE, kernel, 2)
    globe_prior = cv2.morphologyEx(globe_prior, cv2.MORPH_OPEN,  kernel, 1)

    # Reduce false positives by restricting to the central area
    central = np.zeros_like(globe_prior)
    cx_min, cx_max = int(0.05 * W), int(0.95 * W)
    cy_min, cy_max = int(0.05 * H), int(0.95 * H)
    central[cy_min:cy_max, cx_min:cx_max] = 255
    globe_prior = cv2.bitwise_and(globe_prior, central)

    # Seed from the prior
    ys, xs = np.where(globe_prior > 0)
    if len(xs) > 0:
        seed_x = int(np.mean(xs))
        seed_y = int(np.mean(ys))
    else:
        seed_x, seed_y = W // 2, H // 2

    prior_bin = (globe_prior > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        prior_bin, connectivity=8
    )

    # Connected component containing the seed
    label_at_seed = labels[seed_y, seed_x] if 0 <= seed_y < H and 0 <= seed_x < W else 0

    region_mask = np.zeros((H, W), np.uint8)
    if label_at_seed != 0:
        region_mask[labels == label_at_seed] = 255
    else:
        region_mask = globe_prior.copy()

    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, 2)

    # Fit circle to get a smooth mask
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        overlay_rgb = src_rgb.copy()
        final_mask = np.zeros((H, W), np.uint8)
        info = {"seed_point": (seed_x, seed_y), "circle": None}
        return overlay_rgb, final_mask, info

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        cx, cy = seed_x, seed_y
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    radius = np.sqrt(area / np.pi) * 0.97
    cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(radius))

    final_mask = np.zeros((H, W), np.uint8)
    cv2.circle(final_mask, (cx_i, cy_i), r_i, 255, -1)

    final_mask = cv2.bitwise_and(final_mask, globe_prior)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, 1)

    overlay_bgr = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(overlay_bgr, (cx_i, cy_i), r_i, (0, 0, 255), 3)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    info = {
        "circle_center": (cx, cy),
        "circle_radius": radius,
        "area_region": area,
    }

    return overlay_rgb, final_mask, info

def preprocess_for_globe(src_rgb):

    # Normalise illumination and smooth to stabilise colour thresholds
    lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge((L_eq, A, B))
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    rgb_blur = cv2.GaussianBlur(rgb_eq, (7, 7), 0)
    return rgb_blur

WS_T = 146         
KM_INTENSITY = 40   
KM_CLUSTERS = 5     

def combined_segmentation(bgr_img):
   
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    ws_overlay, ws_mask = watershed(bgr_img.copy(), WS_T, False)
    rg_overlay, rg_mask, _ = segment_globe_regiongrow_shape_param(rgb_img.copy(), 170, plot_process=False)
    km_overlay, km_mask = kmeans_segmentation(bgr_img.copy(), KM_INTENSITY, KM_CLUSTERS, False)

    ws_bin = ws_mask > 0
    rg_bin = rg_mask > 0
    km_bin = km_mask > 0

    vote_sum = ws_bin.astype(np.int32) * (rg_bin.astype(np.int32) + km_bin.astype(np.int32))
    comb_mask = (vote_sum >= 1).astype(np.uint8) * 255

    overlay_comb = cv2.cvtColor(bgr_img.copy(), cv2.COLOR_BGR2RGB)
    clean_mask = np.zeros_like(comb_mask)
    contours, _ = cv2.findContours(comb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        if len(hull) > 4:
            ellipse = cv2.fitEllipse(hull)
            (center, axes, angle) = ellipse
            (MA, ma) = axes
            aspect_ratio = min(MA, ma) / max(MA, ma)
            if aspect_ratio > 0.3: 
                # Draw the perfect ellipse
                cv2.ellipse(overlay_comb, ellipse, (255, 0, 0), 2)
                cv2.ellipse(clean_mask, ellipse, 255, -1)
            else:
                # Fallback: Just draw the hull if the ellipse fit was weird
                cv2.drawContours(overlay_comb, [hull], -1, (255, 0, 0), 2)
                cv2.drawContours(clean_mask, [hull], -1, 255, -1)
        else:
            print('else')
            cv2.drawContours(overlay_comb, [cnt], -1, (255, 0, 0), 3)  # red outline

    return overlay_comb, clean_mask, ws_mask, rg_mask, km_mask