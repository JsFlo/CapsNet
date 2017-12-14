package fhc.tfsandbox.capsnettweak.models

data class Capsule(val cap_id: Int, val prediction_row: Int,
                   val paramArray: FloatArray)

data class Prediction(val prediction_row: Int, val real_digit: Int, val imageByteArray: ByteArray)

data class PredictionRow(val capsules: List<Capsule>)

data class ShapeDimensions(val intArray: IntArray)