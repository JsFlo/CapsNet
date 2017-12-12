package fhc.tfsandbox.capsnettweak.capsule_select

import android.support.v7.widget.RecyclerView
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import fhc.tfsandbox.capsnettweak.R
import fhc.tfsandbox.capsnettweak.models.PredictionRow
import kotlinx.android.synthetic.main.item_prediction_row.view.*

class PredictionRowAdapter(val predictionRows: List<PredictionRow>, private val listener: PredictionRowAdapterListener)
    : RecyclerView.Adapter<PredictionRowViewHolder>(), PredictionRowViewHolder.PredictionRowVhListener {

    interface PredictionRowAdapterListener {
        fun onPredictionRowClicked(predictionRow: PredictionRow)
    }

    override fun onCreateViewHolder(parent: ViewGroup?, viewType: Int): PredictionRowViewHolder {
        val itemView = LayoutInflater.from(parent?.context).inflate(R.layout.item_prediction_row, parent, false)
        return PredictionRowViewHolder(itemView, this)
    }

    override fun onBindViewHolder(holder: PredictionRowViewHolder?, position: Int) {
        holder?.onBindViewHolder(predictionRows[position])
    }

    override fun onItemClicked(adapterPosition: Int) {
        listener.onPredictionRowClicked(predictionRows[adapterPosition])
    }

    override fun getItemCount() = predictionRows.size

}

class PredictionRowViewHolder(itemView: View, private val listener: PredictionRowVhListener) : RecyclerView.ViewHolder(itemView), View.OnClickListener {

    interface PredictionRowVhListener {
        fun onItemClicked(adapterPosition: Int)
    }

    init {
        itemView.setOnClickListener(this)
    }

    // TODO: Move this to a presenter
    fun onBindViewHolder(predictionRow: PredictionRow) {
        itemView.pr_real_digit.text = predictionRow.capsules[0].real_digit.toString()
    }

    override fun onClick(v: View?) {
        listener.onItemClicked(adapterPosition)
    }

}