package fhc.tfsandbox.capsnettweak.capsule_tweak

import android.support.v7.widget.RecyclerView
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import fhc.tfsandbox.capsnettweak.R
import fhc.tfsandbox.capsnettweak.models.Capsule
import kotlinx.android.synthetic.main.item_capsule_param.view.*

class CapsuleParamAdapter(private val capsule: Capsule)
    : RecyclerView.Adapter<CapsuleParamViewHolder>(), CapsuleParamViewHolder.CapsuleParamListener {

    fun getUpdatedCapsule(): Capsule {
        return capsule.copy(paramArray = capsuleParams.toFloatArray())
    }

    // might need a clone here ? I don't want to modify the original Capsule
    private val capsuleParams = capsule.paramArray.clone().toMutableList()

    override fun onCreateViewHolder(parent: ViewGroup?, viewType: Int): CapsuleParamViewHolder {
        val itemView = LayoutInflater.from(parent?.context).inflate(R.layout.item_capsule_param, parent, false)
        return CapsuleParamViewHolder(itemView, this)
    }

    override fun onBindViewHolder(holder: CapsuleParamViewHolder?, position: Int) {
        holder?.onBindViewHolder(position, capsuleParams[position])
    }

    override fun getItemCount(): Int = capsuleParams.size

    override fun onCapsuleParamChanged(adapterPosition: Int, newValue: Int) {
        capsuleParams[adapterPosition] = newValue / 100f
    }

}

class CapsuleParamViewHolder(itemView: View, private val listener: CapsuleParamListener) : RecyclerView.ViewHolder(itemView), SeekBar.OnSeekBarChangeListener {
    interface CapsuleParamListener {
        fun onCapsuleParamChanged(adapterPosition: Int, newValue: Int)
    }

    // because total 200 (first 100 for negative)
    private val POSITIVE_NUMBER_OFFSET = 100

    init {
        // 100 for negative 100 for positive
        itemView.cp_seekbar.max = 200
    }

    fun onBindViewHolder(listPosition: Int, rawInput: Float) {
        itemView.cp_param_idx.text = listPosition.toString()
        itemView.cp_updated_value.text = rawInput.toString()

        // don't tell anyone I'm about to update you
        itemView.cp_seekbar.setOnSeekBarChangeListener(null)
        itemView.cp_seekbar.progress = getProgress(rawInput)
        itemView.cp_seekbar.setOnSeekBarChangeListener(this)
    }

    // converts 0.0 - 1.0 into 0 - 200
    // TODO: color needs to go from mid
    private fun getProgress(rawInput: Float): Int {
        val hundredBasedInput = (rawInput * 100).toInt()
        // +
        return if (rawInput > 0) {
            POSITIVE_NUMBER_OFFSET + hundredBasedInput
        } else { // -
            100 - hundredBasedInput
        }
    }

    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {}

    override fun onStartTrackingTouch(seekBar: SeekBar?) {}

    override fun onStopTrackingTouch(seekBar: SeekBar?) {
        seekBar?.let {
            listener.onCapsuleParamChanged(adapterPosition, it.progress)
        }
    }
}