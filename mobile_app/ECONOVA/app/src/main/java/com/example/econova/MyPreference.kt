package com.example.econova

import android.content.Context

val PREFERENCE_NAME = "SharedPreferenceExample"
val PREFERENCE_LANGUAGE = "Language"

class MyPreference(context: Context) {

    // SharedPreferences instance to manage app preferences
    private val preference = context.getSharedPreferences(PREFERENCE_NAME, Context.MODE_PRIVATE)

    // Function to retrieve the saved language, defaulting to "en" if not set
    fun getLanguage(): String {
        return preference.getString(PREFERENCE_LANGUAGE, "en") ?: "en"  // Ensure a non-null return value
    }

    // Function to save the selected language in preferences
    fun setLanguage(language: String) {
        val editor = preference.edit()
        editor.putString(PREFERENCE_LANGUAGE, language)  // Save the language value
        editor.apply()  // Apply changes asynchronously
    }
}
