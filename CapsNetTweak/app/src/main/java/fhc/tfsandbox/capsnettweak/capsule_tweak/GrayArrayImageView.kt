package fhc.tfsandbox.capsnettweak.capsule_tweak

import android.content.Context
import android.graphics.Color
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class GrayArrayImageView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {


    companion object {
        val WIDTH = 28
        val HEIGHT = 28
    }

    private val mOffscreenBitmap: Bitmap
    private val mOffscreenCanvas: Canvas
    private val mPaint = Paint()

    private val mInvMatrix = Matrix()
    private val mMatrix: Matrix by lazy {
        val matrix = Matrix()

        // View size
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        // Model (bitmap) size
        val modelWidth = WIDTH.toFloat()
        val modelHeight = HEIGHT.toFloat()

        val scaleW = viewWidth / modelWidth
        val scaleH = viewHeight / modelHeight

        var scale = scaleW
        if (scale > scaleH) {
            scale = scaleH
        }

        val newCx = modelWidth * scale / 2
        val newCy = modelHeight * scale / 2
        val dx = viewWidth / 2 - newCx
        val dy = viewHeight / 2 - newCy


        matrix.setScale(scale, scale)
        matrix.postTranslate(dx, dy)
        matrix.invert(mInvMatrix)
        matrix
    }

    var dataFloatArray: FloatArray? = null

    init {
        mOffscreenBitmap = Bitmap.createBitmap(28, 28, Bitmap.Config.RGB_565)
        mOffscreenCanvas = Canvas(mOffscreenBitmap)
    }

    fun setArray(floatArray: FloatArray) {
        dataFloatArray = floatArray
        invalidate()
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        canvas?.let {

            for (h in 0 until HEIGHT) {
                for (w in 0 until WIDTH) {
                    dataFloatArray?.let {
                        // data comes in from 0 (white) to 1.f (black)
                        // TODO: fix this hacky way to go from 0 - 1.0 to rgb
                        val pixelColor: Int = 255 - (it[w + (h * WIDTH)] * 255).toInt()
                        mPaint.color = Color.rgb(pixelColor, pixelColor, pixelColor)
                        mOffscreenCanvas.drawPoint(0f + w, 0f + h, mPaint)
                    }
                }
            }

            it.drawBitmap(mOffscreenBitmap, mMatrix, mPaint)
        }
    }
}