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
data class PredictionRow(val capsules: List<Capsule>) : Parcelable {
    // A bit hacky but the data comes in that way (all capsules have a realDigit field)
    // but all capsules in this prediction have the same digit (repeated data)
    // it would be better to catch this earlier OR
    // I should update the db schema!
    // TODO: ^
    val realDigit: Int
        get() = capsules[0].real_digit
}

data class ShapeDimensions(val intArray: IntArray)