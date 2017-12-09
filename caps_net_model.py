import primary_capsules as PrimaryCapsules
import digit_capsules as DigitCapsules


def get_model_output(input_image_batch, batch_size):
    primaryCapsules = PrimaryCapsules.get_primary_capsules(input_image_batch)

    digitCaps_postRouting = DigitCapsules.get_digit_caps_output(primaryCapsules, batch_size)
    return digitCaps_postRouting
