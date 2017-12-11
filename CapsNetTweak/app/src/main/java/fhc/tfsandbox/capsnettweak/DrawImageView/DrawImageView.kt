package fhc.tfsandbox.capsnettweak.DrawImageView

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

interface DrawableImage {
    fun init(model: CanvasModel)
    fun clear()
    fun getBitmap(): Bitmap
}

class DrawImageView(context: Context?, attrs: AttributeSet?) : View(context, attrs), View.OnTouchListener, DrawableImage {

    private var mLastX = 0f
    private var mLastY = 0f
    private val mPaint = Paint()

    private lateinit var mModel: CanvasModel
    private lateinit var mOffscreenBitmap: Bitmap
    private lateinit var mOffscreenCanvas: Canvas

    private val mMatrix: Matrix by lazy {
        val matrix = Matrix()

        // View size
        val width = width.toFloat()
        val height = height.toFloat()

        // Model (bitmap) size
        val modelWidth = mModel.width.toFloat()
        val modelHeight = mModel.height.toFloat()

        val scaleW = width / modelWidth
        val scaleH = height / modelHeight

        var scale = scaleW
        if (scale > scaleH) {
            scale = scaleH
        }

        val newCx = modelWidth * scale / 2
        val newCy = modelHeight * scale / 2
        val dx = width / 2 - newCx
        val dy = height / 2 - newCy


        matrix.setScale(scale, scale)
        matrix.postTranslate(dx, dy)
        matrix.invert(mInvMatrix)
        matrix
    }

    private val mInvMatrix = Matrix()
    private var mDrawnLineSize = 0
    private var initialized = false
    private val tempPointPair = FloatArray(2)

    override fun init(model: CanvasModel) {
        mModel = model
        initialized = true
        mPaint.isAntiAlias = true
        val cm = ColorMatrix()
        cm.setSaturation(0f)
        val f = ColorMatrixColorFilter(cm)
        mPaint.setColorFilter(f)
        mOffscreenBitmap = Bitmap.createBitmap(mModel.width, mModel.height, Bitmap.Config.RGB_565)
        mOffscreenCanvas = Canvas(mOffscreenBitmap)
        mPaint.color = Color.WHITE
        mOffscreenCanvas.drawRect(Rect(0, 0, mModel.width, mModel.height), mPaint)
        mPaint.color = Color.BLACK
        setOnTouchListener(this)
    }

    override fun clear() {
        mModel.lines.clear()
        mDrawnLineSize = 0
        mPaint.color = Color.WHITE
        mOffscreenCanvas.drawRect(Rect(0, 0, mModel.width, mModel.height), mPaint)
        mPaint.color = Color.BLACK
        invalidate()
    }

    public override fun onDraw(canvas: Canvas) {
        if (initialized) {
            var startIndex = mDrawnLineSize - 1
            if (startIndex < 0) {
                startIndex = 0
            }
            // draw the lines in the model starting with index
            // draw to an offScreenCanvas that draws on a bitmap
            mModel.draw(mOffscreenCanvas, mPaint, startIndex)

            // Now draw the bitmap on this canvas
            canvas.drawBitmap(mOffscreenBitmap, mMatrix, mPaint)

            mDrawnLineSize = mModel.lines.size

        }
    }

    // screen pos to local, mutates tempPointPair :(
    fun mapAndUpdateTempPointPair(x: Float, y: Float) {
        tempPointPair[0] = x
        tempPointPair[1] = y
        mInvMatrix.mapPoints(tempPointPair)
    }

    override fun onTouch(p0: View?, event: MotionEvent?): Boolean {
        event?.let {
            val action = event.getAction() and MotionEvent.ACTION_MASK

            if (action == MotionEvent.ACTION_DOWN) {
                processTouchDown(event)
                return true

            } else if (action == MotionEvent.ACTION_MOVE) {
                processTouchMove(event)
                return true

            } else if (action == MotionEvent.ACTION_UP) {
                processTouchUp()
                return true
            }
        }
        return false
    }

    private fun processTouchDown(event: MotionEvent) {
        mLastX = event.x
        mLastY = event.y
        mapAndUpdateTempPointPair(mLastX, mLastY)
        mModel.startLine(tempPointPair[0], tempPointPair[1])
    }

    private fun processTouchMove(event: MotionEvent) {
        mLastX = event.x
        mLastY = event.y
        mapAndUpdateTempPointPair(mLastX, mLastY)
        mModel.addLineSegmentToCurrentLine(tempPointPair[0], tempPointPair[1])

        invalidate()
    }

    private fun processTouchUp() {
        mModel.endLine()
    }

    override fun getBitmap(): Bitmap = mOffscreenBitmap
}