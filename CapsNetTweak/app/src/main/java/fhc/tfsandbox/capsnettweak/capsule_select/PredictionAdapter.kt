package fhc.tfsandbox.capsnettweak.capsule_select

import android.support.v7.widget.RecyclerView
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import fhc.tfsandbox.capsnettweak.R
import fhc.tfsandbox.capsnettweak.models.Prediction
import kotlinx.android.synthetic.main.item_prediction_row.view.*
import android.graphics.BitmapFactory
import android.graphics.Bitmap


class PredictionAdapter(private val predictions: List<Prediction>, private val listener: PredictionAdapterListener)
    : RecyclerView.Adapter<PredictionViewHolder>(), PredictionViewHolder.PredictionVhListener {

    interface PredictionAdapterListener {
        fun onPredictionClicked(prediction: Prediction)
    }

    override fun onCreateViewHolder(parent: ViewGroup?, viewType: Int): PredictionViewHolder {
        val itemView = LayoutInflater.from(parent?.context).inflate(R.layout.item_prediction_row, parent, false)
        return PredictionViewHolder(itemView, this)
    }

    override fun onBindViewHolder(holder: PredictionViewHolder?, position: Int) {
        holder?.onBindViewHolder(predictions[position])
    }

    override fun onItemClicked(adapterPosition: Int) {
        listener.onPredictionClicked(predictions[adapterPosition])
    }

    override fun getItemCount() = predictions.size

}

class PredictionViewHolder(itemView: View, private val listener: PredictionVhListener) : RecyclerView.ViewHolder(itemView), View.OnClickListener {

    interface PredictionVhListener {
        fun onItemClicked(adapterPosition: Int)
    }

    init {
        itemView.setOnClickListener(this)
    }

    // TODO: Move this to a presenter
    fun onBindViewHolder(prediction: Prediction) {
        itemView.pr_real_digit.text = prediction.real_digit.toString()
        val bmp = BitmapFactory.decodeByteArray(prediction.imageByteArray, 0, prediction.imageByteArray.size)
        itemView.pr_source_image.setImageBitmap(bmp)
    }

    override fun onClick(v: View?) {
        listener.onItemClicked(adapterPosition)
    }

}