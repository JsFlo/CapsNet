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

    override fun onCapsuleParamChanged(adapterPosition: Int, newValue: Float) {
        capsuleParams[adapterPosition] = newValue
    }

}

class CapsuleParamViewHolder(itemView: View, private val listener: CapsuleParamListener)
    : RecyclerView.ViewHolder(itemView), CustomValuePicker.CustomValuePickerListener {

    interface CapsuleParamListener {
        fun onCapsuleParamChanged(adapterPosition: Int, newValue: Float)
    }

    init {
        itemView.cp_value_picker.setListener(this)
    }

    fun onBindViewHolder(listPosition: Int, rawInput: Float) {
        itemView.cp_param_idx.text = listPosition.toString()
        // don't tell anyone I'm about to update you
        itemView.cp_value_picker.setProgress(rawInput, true)
    }

    override fun onValueUpdated(negativeOneToOneFloat: Float) {
        listener.onCapsuleParamChanged(adapterPosition, negativeOneToOneFloat)
    }

}