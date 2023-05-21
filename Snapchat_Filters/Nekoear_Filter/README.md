# NEKOEAR FILTER

A dynamic cat ear filter, having the shaking effect when eye-blinks are detected.

# SIMPLE IMPLEMENTATION

We detect the eye-blinks by measuring the eye height and comparing it with its records in previous frames to decide whether the eye is closed or opened in current frame. Then according to the detection result, the ear-shaking effect is implemented by resizing the height of the masking picture.

## AUTHOR

[AyaseNana](https://github.com/NKNaN?tab=repositories)