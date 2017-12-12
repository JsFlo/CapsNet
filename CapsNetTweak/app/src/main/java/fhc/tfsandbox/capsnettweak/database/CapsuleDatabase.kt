package fhc.tfsandbox.capsnettweak.database

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.support.v4.app.ActivityCompat
import fhc.tfsandbox.capsnettweak.capsule_select.debugPrint
import fhc.tfsandbox.capsnettweak.models.Capsule
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
    Table digit_caps
    (cap_id INTEGER, prediction_row INTEGER, real_digit INTEGER,
    param_0 real, param_1 real,param_2 real, param_3 real, param_4 real,param_5 real, param_6 real, param_7 real,
    param_8 real, param_9 real,param_10 real, param_11 real, param_12 real,param_13 real, param_14 real, param_15 real)'''
     */
    private fun getAllCapsulesFromDb(): List<Capsule> {
        val query = "select * From " + TABLE_NAME_DIGIT_CAPS
        val cursor = writableDatabase.rawQuery(query, null)

        val capsules = mutableListOf<Capsule>()
        val floatArrayOffset = 3
        if (cursor.count > 0) {
            if (cursor.moveToFirst()) {
                do {
                    // Should make this a loop that can pull out a variable number of floats out
                    // or some sql skill I don't have
                    // TODO: Look up best way to do this

                    val paramArray = floatArrayOf(cursor.getFloat(floatArrayOffset + 0), cursor.getFloat(floatArrayOffset + 1), cursor.getFloat(floatArrayOffset + 2), cursor.getFloat(floatArrayOffset + 3),
                            cursor.getFloat(floatArrayOffset + 4), cursor.getFloat(floatArrayOffset + 5), cursor.getFloat(floatArrayOffset + 6), cursor.getFloat(floatArrayOffset + 7),
                            cursor.getFloat(floatArrayOffset + 8), cursor.getFloat(floatArrayOffset + 9), cursor.getFloat(floatArrayOffset + 10), cursor.getFloat(floatArrayOffset + 11),
                            cursor.getFloat(floatArrayOffset + 12), cursor.getFloat(floatArrayOffset + 13), cursor.getFloat(floatArrayOffset + 14), cursor.getFloat(floatArrayOffset + 15))

                    "from db paramArray: ${paramArray.size}".debugPrint()
                    capsules.add(Capsule(cap_id = cursor.getInt(0), prediction_row = cursor.getInt(1), real_digit = cursor.getInt(2), paramArray = paramArray))
                } while (cursor.moveToNext())
            }
        }
        return capsules
    }

    fun getAllPredictionRows(): List<PredictionRow> {
        val capsules = getAllCapsulesFromDb()
        return capsules.groupBy { it.prediction_row }.map { PredictionRow(it.value) }
    }

    override fun onCreate(db: SQLiteDatabase?) {}

    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {}

}