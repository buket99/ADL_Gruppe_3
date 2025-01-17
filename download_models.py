import os
import requests
import zipfile

def download_and_test_model(url, download_path, extract_path):
    try:
        # Step 1: Download the file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        with open(download_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded file to {download_path}")

        # Step 2: Check if the file exists and is not empty
        if not os.path.exists(download_path) or os.path.getsize(download_path) == 0:
            print("Error: Downloaded file is empty or does not exist.")
            return

        # Step 3: Verify if it's a valid ZIP file
        if not zipfile.is_zipfile(download_path):
            print("Error: The downloaded file is not a valid ZIP archive.")
            return

        print("The downloaded file is a valid ZIP archive.")

        # Step 4: Extract the ZIP file
        print(f"Extracting ZIP file to {extract_path}...")
        os.makedirs(extract_path, exist_ok=True)  # Ensure extraction path exists
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction complete. Contents extracted to {extract_path}")

        # Step 5: List extracted contents
        print("Extracted files:")
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                print(os.path.join(root, file))

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP archive.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Define the URL, download path, and extraction path
    url = "https://syncandshare.lrz.de/dl/fiRQrzQkL94w4DWzgSNPX/model.dir"
    download_path = "classifiers/model/model.zip"
    extract_path = "classifiers/model"

    # Call the download and test function
    download_and_test_model(url, download_path, extract_path)