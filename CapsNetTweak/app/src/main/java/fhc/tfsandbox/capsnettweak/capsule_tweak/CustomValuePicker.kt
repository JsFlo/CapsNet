package fhc.tfsandbox.capsnettweak.capsule_tweak

import android.content.Context
import android.util.AttributeSet
import android.widget.FrameLayout
import android.widget.SeekBar
import fhc.tfsandbox.capsnettweak.R
import kotlinx.android.synthetic.main.view_value_picker.view.*

/**
 * Wrapper for seekbar with max set to 200
 * First 100 is for negative numbers from: -1.0 -> 0.0
 * Second 100 is the positive number from: 0.0 -> 1.0
 *
 * -1.0 = 0
 * -.5 = 50
 * 0 = 100
 * .5 = 150
 * 1 = 200
 */
class CustomValuePicker(context: Context?, attrs: AttributeSet?) : FrameLayout(context, attrs), SeekBar.OnSeekBarChangeListener {

    interface CustomValuePickerListener {
        fun onValueUpdated(negativeOneToOneFloat: Float)
        fun onStopTrackingTouch()
    }

    companion object {
        const private val POSITIVE_NUMBER_OFFSET = 100
        const private val MAX = 200
    }

    private var listener: CustomValuePickerListener? = null

    init {
        inflate(context, R.layout.view_value_picker, this)
        vp_seekbar.max = MAX
        vp_seekbar.setOnSeekBarChangeListener(this)
    }

    fun setListener(listener: CustomValuePickerListener) {
        this.listener = listener
    }

    fun setProgress(rawValue: Float, setValueSilently: Boolean = false) {
        if (setValueSilently) {
            vp_seekbar.setOnSeekBarChangeListener(null)
        }
        vp_seekbar.progress = convertFloatTo200Int(rawValue)
        if (setValueSilently) {
            vp_seekbar.setOnSeekBarChangeListener(this)
        }

        // set the real raw value
        vp_updated_value.text = rawValue.toString()
    }

    override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
        onProgressChanged(progress)
    }

    override fun onStartTrackingTouch(seekBar: SeekBar?) {

    }

    override fun onStopTrackingTouch(seekBar: SeekBar?) {
        seekBar?.let {
            onProgressChanged(it.progress)
            listener?.onStopTrackingTouch()
        }
    }

    private fun onProgressChanged(progress: Int) {
        val convertedFloat = convert200IntToNegativeOneToOne(progress)
        listener?.onValueUpdated(convertedFloat)
        vp_updated_value.text = convertedFloat.toString()
    }

    private fun convert200IntToNegativeOneToOne(twoHundredBasedValue: Int): Float {
        // 200(1.0) - 100 = 100/100f = 1.0
        // 90(-.10) - 100 = -10/100f = -.10
        val oneHundredBased = twoHundredBasedValue - POSITIVE_NUMBER_OFFSET
        return oneHundredBased / 100f
    }

    private fun convertFloatTo200Int(rawValue: Float): Int {
        return if (rawValue > 0) {
            POSITIVE_NUMBER_OFFSET + (rawValue * 100).toInt()
        } else {
            (rawValue * 100).toInt() + POSITIVE_NUMBER_OFFSET
        }
    }
}