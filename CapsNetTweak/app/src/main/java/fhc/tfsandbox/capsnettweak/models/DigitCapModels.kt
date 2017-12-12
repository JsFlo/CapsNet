package fhc.tfsandbox.capsnettweak.models

import android.annotation.SuppressLint
import android.gesture.Prediction
import android.os.Parcelable
import kotlinx.android.parcel.Parcelize

@SuppressLint("ParcelCreator")
@Parcelize
data class Capsule(val cap_id: Int, val prediction_row: Int, val real_digit: Int,
                   val paramArray: FloatArray) : Parcelable


@SuppressLint("ParcelCreator")
@Parcelize
data class PredictionRow(val capsules: List<Capsule>) : Parcelable


data class Predictions(val predictionRows: Prediction)