import ee 




def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)

# 1. Define a function to calculate footprint area and add it as a property
def add_area(image):
    """Adds an area column to an ee.Image, usful for determining if the image is complete or not.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: Image with the aree calculated in km2, saved to a column named 'footprint_area'
    """
    area = image.geometry().area().divide(1e6)  # Area in km²
    return image.set('footprint_area', area)


def invertCloudProbabilityBand(image):
    """Estimates pixel-level quality based on Aerosol Optical Thickness.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: Image with quality parameter included.
    """
    cldProb = image.select('AOT')
    cldProbInv = cldProb.multiply(-1).rename('quality')
    return image.addBands(cldProbInv)


