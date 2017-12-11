package fhc.tfsandbox.capsnettweak

import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.util.Log
import fhc.tfsandbox.capsnettweak.classifier.NodeDef
import org.tensorflow.contrib.android.TensorFlowInferenceInterface
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val outputNodeDef = NodeDef("output", FloatArray(784).toTypedArray())

        val tfInference = TensorFlowInferenceInterface(assets, "model_graph.pb")

//        // 16 dimensions
//        val caps1: FloatArray = floatArrayOf(10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
//                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f)
//        val caps2: FloatArray = caps1
//        val caps3: FloatArray = caps1
//        val caps4: FloatArray = caps1
//        val caps5: FloatArray = caps1
//        val caps6: FloatArray = caps1
//        val caps7: FloatArray = caps1
//        val caps8: FloatArray = caps1
//        val caps9: FloatArray = caps1
//        val caps10: FloatArray = caps1
//
//        val row: Array<FloatArray> = arrayOf(caps1, caps2, caps3, caps4, caps5, caps6, caps7, caps8, caps9, caps10)
//
//        val input: Array<Array<FloatArray>> = arrayOf(row)

        val input = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
//                         [[-0.19408944]
//                        [-0.2736159 ]
//                        [ 0.27231807]
//                        [-0.22223102]
//                        [ 0.31453183]
//                        [-0.06305426]
//                        [ 0.14150277]
//                        [-0.27442247]

//                        [ 0.2606332 ]
//                        [-0.07740127]
//                        [-0.23510878]
//                        [-0.13882922]
//                        [-0.28842384]
//                        [ 0.38337508]
//                        [ 0.1068143 ]
//                        [ 0.25122228]]
                -0.19408944f, -0.2736159f, 0.27231807f, -0.22223102f, 0.31453183f, -0.06305426f, 0.14150277f, -0.27442247f,
                0.2606332f, -0.07740127f, -0.23510878f, -0.13882922f, -0.28842384f, 0.38337508f, 0.1068143f, 0.25122228f, // 8

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,

                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        Log.d("test", "${input.size}")
        val floatBuffer = FloatBuffer.wrap(input, 0, 160)


        // feed
        tfInference.feed("input:0", floatBuffer, *reshape(intArrayOf(1, 1, 10, 16, 1).toTypedArray().toIntArray()))

        // fetch
        val floatOutputs = outputNodeDef.shape.toFloatArray()
        tfInference.run(arrayOf(outputNodeDef.name))
        tfInference.fetch("output", floatOutputs)
        floatOutputs.forEach { Log.d("test", "$it") }

        //Bitmap.createBitmap(28,28, Bitmap.Config.ALPHA_8)
        // You are using RGBA that's why Config is ARGB.8888
//        val bitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.ARGB_8888);
//        // vector is your int[] of ARGB
//        bitmap.copyPixelsFromBuffer(FloatBuffer.wrap(floatOutputs));
//        image_view.setImageBitmap(bitmap)

        // make image from float outputs
    }

    private fun reshape(intArray: IntArray) = intArray.map { it.toLong() }.toLongArray()

    private fun reshape(inputArray: Array<Array<FloatArray>>): FloatArray {
        return arrayOf(10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f,

                10f, 2f, 34f, 45f, 52f, 62f, 70f, 8f,
                84f, 75f, 62f, 52f, 40f, 3f, 24f, 15f).toFloatArray()
    }
}
