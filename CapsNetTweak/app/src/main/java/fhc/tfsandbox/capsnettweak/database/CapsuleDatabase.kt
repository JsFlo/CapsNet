package fhc.tfsandbox.capsnettweak.database

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.support.v4.app.ActivityCompat
import fhc.tfsandbox.capsnettweak.capsule_select.debugPrint
import fhc.tfsandbox.capsnettweak.models.Capsule
import fhc.tfsandbox.capsnettweak.models.Prediction
import fhc.tfsandbox.capsnettweak.models.PredictionRow
import java.io.File
import java.io.FileOutputStream

// TODO: See if "Room" plays nice with the copied over
class CapsuleDatabase(context: Context, databaseName: String, databaseVersion: Int)
    : SQLiteOpenHelper(context, databaseName, null, databaseVersion) {

    companion object {
        // Storage Permissions variables
        private val REQUEST_EXTERNAL_STORAGE = 1
        private val PERMISSIONS_STORAGE = arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)

        private val ANDROID_DB_PATH = "/data/data/fhc.tfsandbox.capsnettweak/databases/"
        private val DB_NAME = "minimal_decoder.db"
        private val DB_VERSION = 1

        private val TABLE_NAME_DIGIT_CAPS = "digit_caps"
        private val TABLE_NAME_PREDICTIONS = "prediction"

        // assumes it has permissions
        fun getCapsuleDatabase(context: Context): CapsuleDatabase {
            if (!doesDbExist()) {
                copyDbFromAssetsToAndroidDb(context, DB_NAME, ANDROID_DB_PATH)
            }
            return CapsuleDatabase(context, DB_NAME, DB_VERSION)
        }

        fun needsPermissions(activity: Activity): Boolean = !doesDbExist() && !hasPermissions(activity)

        fun requestPermissions(activity: Activity) {
            // TODO: Handle showing explanation
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE)
        }

        fun onRequestPermissionsResult(context: Context, requestCode: Int, permissions: Array<out String>, grantResults: IntArray): CapsuleDatabase? {
            if (requestCode == REQUEST_EXTERNAL_STORAGE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // permission granted
                return getCapsuleDatabase(context)
            }

            return null
        }

        private fun hasPermissions(activity: Activity): Boolean {
            val writePermission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            val readPermission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE)

            return writePermission == PackageManager.PERMISSION_GRANTED && readPermission == PackageManager.PERMISSION_GRANTED
        }

        private fun copyDbFromAssetsToAndroidDb(context: Context, assetsDbName: String,
                                                androidDbPath: String, androidDbName: String = assetsDbName) {
            File(androidDbPath).mkdirs()
            // open db in assets as stream
            context.assets.open(assetsDbName).use { assets_db_stream ->
                // file output stream for new db
                FileOutputStream("$androidDbPath$androidDbName").use { file_output_stream ->
                    // transfer bytes
                    val buffer = ByteArray(1024)

                    var length: Int
                    length = assets_db_stream.read(buffer)
                    while (length > 0) {
                        file_output_stream.write(buffer, 0, length)
                        length = assets_db_stream.read(buffer)
                    }
                }
            }
        }

        private fun doesDbExist() = File(ANDROID_DB_PATH, DB_NAME).exists()
    }

    /**
     * prediction
    (prediction_row INTEGER, real_digit INTEGER, image_bytes blob)
     */
    fun getPredictions(): List<Prediction> {
        val query = "select * From " + TABLE_NAME_PREDICTIONS
        val cursor = writableDatabase.rawQuery(query, null)
        val predictions = mutableListOf<Prediction>()
        if (cursor.count > 0) {
            if (cursor.moveToFirst()) {
                do {
                    predictions.add(Prediction(
                            prediction_row = cursor.getInt("prediction_row"),
                            real_digit = cursor.getInt("real_digit"),
                            imageByteArray = cursor.getBlob("image_bytes")
                    ))


                } while (cursor.moveToNext())
            }
        }
        return predictions
    }


    /**
    Table digit_caps
    (cap_id INTEGER, prediction_row INTEGER,
    param_0 real, param_1 real,param_2 real, param_3 real, param_4 real,param_5 real, param_6 real, param_7 real,
    param_8 real, param_9 real,param_10 real, param_11 real, param_12 real,param_13 real, param_14 real, param_15 real)'''
     */
    fun getPredictionRow(predictionRow: Int): PredictionRow {
        val query = "select * from digit_caps where digit_caps.prediction_row = ?"
        val cursor = writableDatabase.rawQuery(query, arrayOf(predictionRow.toString()))

        val capsules = mutableListOf<Capsule>()
        if (cursor.count > 0) {
            if (cursor.moveToFirst()) {
                do {
                    val paramArray = floatArrayOf(cursor.getFloat("param_0"), cursor.getFloat("param_1"), cursor.getFloat("param_2"), cursor.getFloat("param_3"),
                            cursor.getFloat("param_4"), cursor.getFloat("param_5"), cursor.getFloat("param_6"), cursor.getFloat("param_7"),
                            cursor.getFloat("param_8"), cursor.getFloat("param_9"), cursor.getFloat("param_10"), cursor.getFloat("param_11"),
                            cursor.getFloat("param_12"), cursor.getFloat("param_13"), cursor.getFloat("param_14"), cursor.getFloat("param_15"))

                    capsules.add(Capsule(
                            cap_id = cursor.getInt("cap_id"),
                            prediction_row = cursor.getInt("prediction_row"),
                            paramArray = paramArray))
                } while (cursor.moveToNext())
            }
        }
        return PredictionRow(capsules)
    }

    override fun onCreate(db: SQLiteDatabase?) {}

    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {}

    fun Cursor.getInt(columnName: String) = getInt(getColumnIndex(columnName))
    fun Cursor.getFloat(columnName: String) = getFloat(getColumnIndex(columnName))
    fun Cursor.getBlob(columnName: String) = getBlob(getColumnIndex(columnName))
}