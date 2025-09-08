package com.example.econova

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.airbnb.lottie.LottieAnimationView

class MainActivity_01 : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main01)

        // Handle edge-to-edge window insets
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Reference the second LottieAnimationView
        val secondAnimation: LottieAnimationView = findViewById(R.id.secondAnimation)

        // Initially make the second animation invisible
        secondAnimation.visibility = View.INVISIBLE

        // Delay the second animation by 3 seconds
        Handler(Looper.getMainLooper()).postDelayed({
            secondAnimation.visibility = View.VISIBLE // Make it visible
            secondAnimation.playAnimation() // Start the animation

            // Set a click listener to navigate to MainActivity when clicked
            secondAnimation.setOnClickListener {
                val intent = Intent(this@MainActivity_01, MainActivity::class.java)
                startActivity(intent) // Navigate to MainActivity
            }

        }, 3000) // 3000 milliseconds = 3 seconds
    }
}
