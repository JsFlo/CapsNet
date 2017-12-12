package fhc.tfsandbox.capsnettweak.common

import fhc.tfsandbox.capsnettweak.models.PredictionRow
import fhc.tfsandbox.capsnettweak.models.ShapeDimensions
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.nio.FloatBuffer


fun TensorFlowInferenceInterface.feed(tensorName: String, predictionRow: PredictionRow, shape: ShapeDimensions) {
    feed(tensorName, reshapePredictionRow(predictionRow), *reshape(shape.intArray))
}

private fun reshape(intArray: IntArray) = intArray.map { it.toLong() }.toLongArray()

private fun reshapePredictionRow(predictionRow: PredictionRow): FloatBuffer {
//    predictionRow.capsules.drop(1).fold(predictionRow.capsules.first().paramArray, { acc: FloatArray, capsule: Capsule -> acc + capsule.paramArray })
    val flattenInput: FloatArray = predictionRow.capsules.map { it.paramArray }.reduce({ acc, floats -> acc + floats })
    return FloatBuffer.wrap(flattenInput, 0, flattenInput.size)
}

fun TensorFlowInferenceInterface.runAndFetch(tensorName: String, outputs: FloatArray): FloatArray {
    run(arrayOf(tensorName))
    fetch(tensorName, outputs)
    return outputs
}