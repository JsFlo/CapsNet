package fhc.tfsandbox.capsnettweak.database

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

public class CapsuleDatabase(context: Context, databaseName: String, databaseVersion: Int) : SQLiteOpenHelper(context, databaseName, null, databaseVersion) {

    companion object {
        private val DB_PATH = "/data/data/fhc.tfsandbox.capsnettweak/databases/"
        private val DATABASE_NAME = "minimal_decoder.db"
        private val DATABASE_VERSION = 1

        private val TABLE_NAME = "digit_caps"

        fun getCapsuleDatabase(context: Context): CapsuleDatabase {
            copyDataBase(context)
            return CapsuleDatabase(context, DATABASE_NAME, DATABASE_VERSION)
        }

        @Throws(IOException::class)
        fun copyDataBase(context: Context) {
            if (!doesDbExist()) {
                File(DB_PATH).mkdirs()
//                File(DB_PATH, DATABASE_NAME).createNewFile()


                //Open your local db as the input stream
                val myInput = context.assets.open(DATABASE_NAME)
                // Path to the just created empty db
                val outFileName = DB_PATH + DATABASE_NAME
                //Open the empty db as the output stream
                val fileOutputStream = FileOutputStream(outFileName)
                //transfer bytes from the input file to the output file
                val buffer = ByteArray(1024)
                var length: Int
                length = myInput.read(buffer)
                while (length > 0) {
                    fileOutputStream.write(buffer, 0, length)
                    length = myInput.read(buffer)
                }

                //Close the streams
                fileOutputStream.flush()
                fileOutputStream.close()
                myInput.close()
            }
        }

        private fun copyDBFromAssetsIfNeeded(context: Context) {
            if (!doesDbExist()) {
                // open db in assets as stream
                context.assets.open(DATABASE_NAME).use { assets_db_stream ->
                    // file output stream for new db
                    FileOutputStream("$DB_PATH$DATABASE_NAME").use { file_output_stream ->
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
        }

        private fun doesDbExist() = File(DB_PATH, DATABASE_NAME).exists()
    }

    fun getUserNameFromDB(): String? {
        val query = "select * From " + TABLE_NAME
        val cursor = writableDatabase.rawQuery(query, null)
        var userName: String? = null
        if (cursor.getCount() > 0) {
            if (cursor.moveToFirst()) {
                do {
                    userName = cursor.getString(0)
                } while (cursor.moveToNext())
            }
        }
        return userName
    }

    override fun onCreate(db: SQLiteDatabase?) {}

    override fun onUpgrade(db: SQLiteDatabase?, oldVersion: Int, newVersion: Int) {}

}