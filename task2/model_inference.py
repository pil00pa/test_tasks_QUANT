import rasterio
import numpy as np
import matplotlib.pyplot as plt


def imread_3bands(path, bands):
    """
    Load and preprocess 3-band satellite imagery.
    
    Parameters:
    path (str): Base path to the Sentinel-2 imagery files.
    bands (list): List of band names (e.g. ["B04", "B03", "B02"]).
    
    Returns:
    numpy.ndarray: 3D array with shape (height, width, 3) containing the 3 preprocessed bands.
    """
    channels = []
    for band in bands:
        with rasterio.open(path + band + '.jp2', "r", driver="JP2OpenJPEG") as src:
            channel = src.read(1).astype(float)
            
            # Perform IQR normalization
            q1 = np.percentile(channel, 25)
            q3 = np.percentile(channel, 75)
            iqr = q3 - q1
            channel /= q3 + 1.5 * iqr
            
            channel_8bit = (np.clip(channel, 0, 1) * 255).astype(np.uint8)
            channels.append(channel_8bit)
    
    # Stack the 3 bands into a single 3D array
    return np.stack(channels, axis=-1)


def visualize_matches(img1, img2, mkpts0, mkpts1, confidence, conf_threshold=0.9, dtype=float):
    """
    Visualize the matched keypoints between two images with connecting lines.
    Works with both color and grayscale images.
    
    Parameters:
    img1 (numpy.ndarray): First input image.
    img2 (numpy.ndarray): Second input image.
    mkpts0 (numpy.ndarray): Keypoints from the first image.
    mkpts1 (numpy.ndarray): Keypoints from the second image.
    confidence (numpy.ndarray): Confidence scores for the matches.
    conf_threshold (float): Minimum confidence threshold for displaying matches.
    dtype (type): Data type for the output image.
    """
    # Filter the matches based on confidence threshold
    mask = confidence > conf_threshold
    mkpts0 = mkpts0[mask]
    mkpts1 = mkpts1[mask]
    confidence = confidence[mask]
    
    # Convert the images to 3-channel (RGB) if they are grayscale
    if len(img1.shape) == 2:  # Grayscale image
        img1_colored = np.stack([img1] * 3, axis=-1)
    else:  # Color image
        img1_colored = img1
    
    if len(img2.shape) == 2:  # Grayscale image
        img2_colored = np.stack([img2] * 3, axis=-1)
    else:  # Color image
        img2_colored = img2
    
    # Get the dimensions of the images
    h1, w1 = img1_colored.shape[:2]
    h2, w2 = img2_colored.shape[:2]
    
    # Create a canvas to place the two images side-by-side
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=dtype)
    canvas[:h1, :w1, :] = img1_colored
    canvas[:h2, w1:w1+w2, :] = img2_colored
    
    fig, ax = plt.subplots(figsize=(13, 26))
    ax.imshow(canvas)
    
    # Draw the matches with connecting lines
    for (x1, y1), (x2, y2), conf in zip(mkpts0, mkpts1, confidence):
        color = plt.cm.viridis(conf)  # Color the line based on confidence
        x2_shifted = x2 + w1  # Shift the x-coordinate for the second image
        ax.plot([x1, x2_shifted], [y1, y2], color=color, linewidth=1)  # Draw the line
        ax.plot(x1, y1, 'o', color=color, markersize=7)  # Draw the point on the first image
        ax.plot(x2_shifted, y2, 'o', color=color, markersize=7)  # Draw the point on the second image
    
    ax.set_title(f'Matches visualized: {len(mkpts0)}')
    ax.axis('off')
    plt.tight_layout()


def prepare_matching(matches_dict, conf_threshold=0.9, target_height=10980, target_width=10980):
    """
    Prepare the matching results for visualization.
    
    Parameters:
    matches_dict (dict): Dictionary containing matching results for each image piece.
    conf_threshold (float): Minimum confidence threshold for including a match.
    target_height (int): Target height of the full-size images.
    target_width (int): Target width of the full-size images.
    
    Returns:
    list: A list containing:
         - numpy.ndarray: Keypoints from the first image.
         - numpy.ndarray: Keypoints from the second image.
         - numpy.ndarray: Confidence scores for the matches.
    """
    # Get the first matching result to determine image sizes
    img1, img2, mkpts0, mkpts1, confidence = list(matches_dict.values())[0]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    prep_matchings = {}
    
    # Iterate over each image piece and process the matches
    for key, (_, _, mkpts0, mkpts1, confidence) in matches_dict.items():
        y_offset, x_offset = key
        
        # Filter matches by confidence threshold and the distance between points
        mask = (confidence > conf_threshold) & (np.linalg.norm(mkpts0 - mkpts1, axis=1) <= 5)
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        confidence = confidence[mask]
        
        # Shift keypoint coordinates to match their position in the full-size image
        mkpts0[:, 0] += x_offset * w1  # Shift x-coordinates for the first image
        mkpts0[:, 1] += y_offset * h1  # Shift y-coordinates for the first image
        mkpts1[:, 0] += x_offset * w2  # Shift x-coordinates for the second image
        mkpts1[:, 1] += y_offset * h2  # Shift y-coordinates for the second image
        
        prep_matchings[key] = [mkpts0, mkpts1, confidence]
    
    mkpts0_all = []
    mkpts1_all = []
    confidence_all = []
    
    for mkpts0, mkpts1, conf in prep_matchings.values():
        mkpts0_all.extend(mkpts0)
        mkpts1_all.extend(mkpts1)
        confidence_all.extend(conf)
    
    mkpts0_all = np.array(mkpts0_all)
    mkpts1_all = np.array(mkpts1_all)
    confidence_all = np.array(confidence_all)
    
    return [mkpts0_all, mkpts1_all, confidence_all]


if __name__ == "main":
    im1_path = "Sentinel2/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510/S2A_MSIL1C_20160212T084052_N0201_R064_T36UYA_20160212T084510.SAFE/GRANULE/L1C_T36UYA_A003350_20160212T084510/IMG_DATA/T36UYA_20160212T084052_B"
    im2_path = "Sentinel2/S2B_MSIL1C_20190412T083609_N0207_R064_T36UYA_20190412T122445/S2B_MSIL1C_20190412T083609_N0207_R064_T36UYA_20190412T122445.SAFE/GRANULE/L1C_T36UYA_A010958_20190412T084433/IMG_DATA/T36UYA_20190412T083609_B"

    # Load precomputed matching results
    results = np.load('results.npy', allow_pickle=True).item()

    good_matchings = prepare_matching(results)
    print("Number of matches:", len(good_matchings[0]))

    some_matchings = []
    choice = np.random.choice(len(good_matchings[0]), size=100, replace=False)
    for data in good_matchings:
        some_matchings.append(data[choice])

    im1_rgb = imread_3bands(path=im1_path, bands=["04", "03", "02"])
    im2_rgb = imread_3bands(path=im2_path, bands=["04", "03", "02"])

    visualize_matches(im1_rgb, im2_rgb, some_matchings[0], some_matchings[1], some_matchings[2], dtype=np.uint8)
