package fhc.tfsandbox.capsnettweak.capsule_tweak

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.support.v7.widget.LinearLayoutManager
import android.view.MenuItem
import android.view.View
import fhc.tfsandbox.capsnettweak.R
import fhc.tfsandbox.capsnettweak.common.feed
import fhc.tfsandbox.capsnettweak.common.runAndFetch
import fhc.tfsandbox.capsnettweak.database.CapsuleDatabase
import fhc.tfsandbox.capsnettweak.models.PredictionRow
import fhc.tfsandbox.capsnettweak.models.ShapeDimensions
import kotlinx.android.synthetic.main.activity_tweak.*
import kotlinx.coroutines.experimental.launch
import org.tensorflow.contrib.android.TensorFlowInferenceInterface

class TweakActivity : AppCompatActivity(), View.OnClickListener {

    companion object {
        const val EXTRA_PREDICTION_ROW_ID = "EXTRA_PREDICTION_ROW"
        const val EXTRA_REAL_DIGIT = "EXTRA_REAL_DIGIT"
        fun newIntent(context: Context, predictionRow: Int, realDigit: Int): Intent {
            val intent = Intent(context, TweakActivity::class.java)
            intent.putExtra(EXTRA_PREDICTION_ROW_ID, predictionRow)
            intent.putExtra(EXTRA_REAL_DIGIT, realDigit)
            return intent
        }
    }

    private lateinit var predictionRow: PredictionRow
    private var realDigit: Int = 0

    private lateinit var tfInference: TensorFlowInferenceInterface

    private lateinit var adapter: CapsuleParamAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tweak)
        title = "Reconstruction"
        supportActionBar?.let {
            it.setHomeButtonEnabled(true)
            it.setDisplayHomeAsUpEnabled(true)
        }

        with(reconstruct_button) {
            isEnabled = false
            setOnClickListener(this@TweakActivity)
        }

        // get extra
        val predictionRowId = savedInstanceState?.getInt(EXTRA_PREDICTION_ROW_ID)
                ?: intent.getIntExtra(EXTRA_PREDICTION_ROW_ID, 0)
        realDigit = savedInstanceState?.getInt(EXTRA_REAL_DIGIT)
                ?: intent.getIntExtra(EXTRA_REAL_DIGIT, 0)


        // get tf inference
        tfInference = TensorFlowInferenceInterface(assets, "model_graph.pb")

        // pull out the 1 capsule that has data
        launch {
            val db = CapsuleDatabase.getCapsuleDatabase(this@TweakActivity)
            predictionRow = db.getPredictionRow(predictionRowId)
            val focusedCapsule = predictionRow.capsules[realDigit]
            runOnUiThread {
                adapter = CapsuleParamAdapter(focusedCapsule)
                tweak_rv.layoutManager = LinearLayoutManager(this@TweakActivity, LinearLayoutManager.VERTICAL, false)
                tweak_rv.adapter = adapter
            }
            runInference(predictionRow)
        }
    }

    suspend private fun runInference(inferencePredictionRow: PredictionRow) {
        tfInference.feed("input:0", inferencePredictionRow, ShapeDimensions(intArrayOf(1, 1, 10, 16, 1)))
        val floatOutputs = FloatArray(784).toTypedArray().toFloatArray()
        tfInference.runAndFetch("output", floatOutputs)

        runOnUiThread {
            image_view.setArray(floatOutputs)
            reconstruct_button.isEnabled = true
        }
    }

    override fun onClick(v: View?) {
        reconstruct_button.isEnabled = false
        val updatedCapsule = adapter.getUpdatedCapsule()

        val updatedPredictionRow = ArrayList(predictionRow.capsules)
        updatedPredictionRow[realDigit] = updatedCapsule
        launch { runInference(PredictionRow(updatedPredictionRow)) }
    }

    override fun onOptionsItemSelected(item: MenuItem?): Boolean {
        item?.let {
            if (it.itemId == android.R.id.home) {
                onBackPressed()
                return true
            }
        }

        return super.onOptionsItemSelected(item)
    }
}
