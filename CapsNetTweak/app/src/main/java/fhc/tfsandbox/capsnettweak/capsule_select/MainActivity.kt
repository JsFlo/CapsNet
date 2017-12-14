package fhc.tfsandbox.capsnettweak.capsule_select

import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.support.v7.widget.GridLayoutManager
import android.util.Log
import fhc.tfsandbox.capsnettweak.R
import fhc.tfsandbox.capsnettweak.capsule_tweak.TweakActivity
import fhc.tfsandbox.capsnettweak.database.CapsuleDatabase
import fhc.tfsandbox.capsnettweak.models.PredictionRow
import kotlinx.android.synthetic.main.activity_main.*

fun String.debugPrint() {
    Log.d("test", this)
}

class MainActivity : AppCompatActivity(), PredictionRowAdapter.PredictionRowAdapterListener {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        title = "Capsule Outputs"
        if (!CapsuleDatabase.needsPermissions(this)) {
            onDbReady(CapsuleDatabase.getCapsuleDatabase(this))
        } else {
            CapsuleDatabase.requestPermissions(this)
        }
    }

    private fun onDbReady(capsuleDatabase: CapsuleDatabase) {
        val capsules = capsuleDatabase.getAllPredictionRows()
        val adapter = PredictionRowAdapter(capsules, this)
        prediction_row_rv.layoutManager = GridLayoutManager(this, 2)
        prediction_row_rv.adapter = adapter
    }

    override fun onPredictionRowClicked(predictionRow: PredictionRow) {
        startActivity(TweakActivity.newIntent(this, predictionRow))
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val db = CapsuleDatabase.onRequestPermissionsResult(this, requestCode, permissions, grantResults)
        db?.let {
            onDbReady(CapsuleDatabase.getCapsuleDatabase(this))
        }
    }
}
