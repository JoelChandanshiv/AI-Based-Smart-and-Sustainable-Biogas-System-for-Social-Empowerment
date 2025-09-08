package com.example.econova

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.airbnb.lottie.LottieAnimationView

class SplashActivity : AppCompatActivity() {

    private lateinit var firstAnimation: LottieAnimationView
    private lateinit var secondAnimation: LottieAnimationView
    private lateinit var leafAnimation: LottieAnimationView // New leaf animation

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_splash)

        // Apply edge-to-edge insets
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Make the activity fullscreen
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )

        // Initialize LottieAnimationView objects
        firstAnimation = findViewById(R.id.lottieAnimationView)
        secondAnimation = findViewById(R.id.transformationanimation)
        leafAnimation = findViewById(R.id.leafani) // Initialize leaf animation

        // Start the first animation after 1 second delay
        Handler(Looper.getMainLooper()).postDelayed({
            firstAnimation.playAnimation()
        }, 1000) // 1 second delay

        // Start the second animation after 2 seconds delay
        Handler(Looper.getMainLooper()).postDelayed({
            secondAnimation.playAnimation()
        }, 2000) // 2 seconds delay

        // Start the leaf animation after 2 seconds delay
        Handler(Looper.getMainLooper()).postDelayed({
            leafAnimation.playAnimation() // Play the leaf animation
        }, 2000) // 2 seconds delay for leaf animation

        // Proceed to MainActivity after 5 seconds
        Handler(Looper.getMainLooper()).postDelayed({
            val intent = Intent(this@SplashActivity, MainActivity_01::class.java)
            startActivity(intent)
            finish() // Close SplashActivity after launching MainActivity
        }, 6100) // 5 seconds delay before navigating to MainActivity
    }
}
