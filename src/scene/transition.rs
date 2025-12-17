//! Transition types and easing functions for scene switching.

use crate::draw2d::Color;

/// Easing functions for smooth transitions.
///
/// These control the acceleration curve of transition animations.
#[derive(Clone, Copy, Debug, Default)]
pub enum Easing {
    /// Constant speed throughout.
    #[default]
    Linear,
    /// Start slow, accelerate.
    EaseIn,
    /// Start fast, decelerate.
    EaseOut,
    /// Start slow, speed up, then slow down.
    EaseInOut,
}

impl Easing {
    /// Apply the easing function to a linear progress value (0.0 to 1.0).
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::EaseIn => t * t,
            Easing::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            Easing::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
        }
    }
}

/// Type of transition effect between scenes.
#[derive(Clone, Debug)]
pub enum TransitionKind {
    /// Instant switch, no visual transition.
    Instant,
    /// Fade to a solid color, then fade in the new scene.
    FadeToColor { color: Color },
    /// Crossfade: blend old scene out while blending new scene in.
    Crossfade,
}

/// Configuration for a scene transition.
///
/// Create transitions using the constructor methods and optionally chain
/// `.easing()` to customize the animation curve.
///
/// # Example
///
/// ```ignore
/// // Simple fade to black over 0.5 seconds
/// Transition::fade_to_black(0.5)
///
/// // Crossfade with custom easing
/// Transition::crossfade(0.8).easing(Easing::EaseInOut)
/// ```
#[derive(Clone, Debug)]
pub struct Transition {
    /// The type of transition effect.
    pub kind: TransitionKind,
    /// Duration in seconds.
    pub duration: f32,
    /// Easing function for the animation.
    pub easing: Easing,
}

impl Default for Transition {
    fn default() -> Self {
        Self::instant()
    }
}

impl Transition {
    /// Create an instant transition (no animation).
    pub fn instant() -> Self {
        Self {
            kind: TransitionKind::Instant,
            duration: 0.0,
            easing: Easing::Linear,
        }
    }

    /// Create a fade-to-black transition.
    ///
    /// The current scene fades to black, then the new scene fades in from black.
    ///
    /// # Arguments
    ///
    /// * `duration` - Total transition duration in seconds (half for fade out, half for fade in)
    pub fn fade_to_black(duration: f32) -> Self {
        Self {
            kind: TransitionKind::FadeToColor {
                color: Color::BLACK,
            },
            duration,
            easing: Easing::EaseInOut,
        }
    }

    /// Create a fade-to-white transition.
    ///
    /// The current scene fades to white, then the new scene fades in from white.
    ///
    /// # Arguments
    ///
    /// * `duration` - Total transition duration in seconds
    pub fn fade_to_white(duration: f32) -> Self {
        Self {
            kind: TransitionKind::FadeToColor {
                color: Color::WHITE,
            },
            duration,
            easing: Easing::EaseInOut,
        }
    }

    /// Create a fade-to-color transition with a custom color.
    ///
    /// # Arguments
    ///
    /// * `color` - The color to fade through
    /// * `duration` - Total transition duration in seconds
    pub fn fade_to_color(color: Color, duration: f32) -> Self {
        Self {
            kind: TransitionKind::FadeToColor { color },
            duration,
            easing: Easing::EaseInOut,
        }
    }

    /// Create a crossfade transition.
    ///
    /// The old scene fades out while the new scene simultaneously fades in.
    /// This requires rendering both scenes during the transition.
    ///
    /// # Arguments
    ///
    /// * `duration` - Total transition duration in seconds
    pub fn crossfade(duration: f32) -> Self {
        Self {
            kind: TransitionKind::Crossfade,
            duration,
            easing: Easing::EaseInOut,
        }
    }

    /// Set the easing function for this transition.
    pub fn easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }

    /// Set the duration for this transition.
    pub fn duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }
}

/// Internal phase of an active transition.
#[derive(Debug, Clone, Copy)]
pub enum TransitionPhase {
    /// Fading out the old scene (progress 0.0 = start, 1.0 = fully faded).
    FadingOut,
    /// At midpoint - ready to swap scenes.
    Midpoint,
    /// Fading in the new scene (progress 0.0 = start, 1.0 = fully visible).
    FadingIn,
    /// Crossfade: both scenes visible, blending (progress 0.0 = old scene, 1.0 = new scene).
    Crossfading,
}

/// Internal state of an active transition.
#[derive(Debug)]
pub struct ActiveTransition {
    /// The transition configuration.
    pub transition: Transition,
    /// Current phase of the transition.
    pub phase: TransitionPhase,
    /// Linear progress within the current phase (0.0 to 1.0).
    pub progress: f32,
    /// Time when the transition started.
    pub start_time: f32,
    /// Target scene ID.
    pub target_scene: String,
    /// Source scene ID (for crossfade).
    pub source_scene: String,
}

impl ActiveTransition {
    /// Create a new active transition.
    pub fn new(
        transition: Transition,
        source_scene: String,
        target_scene: String,
        start_time: f32,
    ) -> Self {
        let phase = match transition.kind {
            TransitionKind::Instant => TransitionPhase::Midpoint,
            TransitionKind::FadeToColor { .. } => TransitionPhase::FadingOut,
            TransitionKind::Crossfade => TransitionPhase::Crossfading,
        };

        Self {
            transition,
            phase,
            progress: 0.0,
            start_time,
            target_scene,
            source_scene,
        }
    }

    /// Update the transition state based on current time.
    ///
    /// Returns `true` if the transition is complete.
    pub fn update(&mut self, current_time: f32) -> bool {
        let elapsed = current_time - self.start_time;

        match self.transition.kind {
            TransitionKind::Instant => {
                self.phase = TransitionPhase::Midpoint;
                true
            }
            TransitionKind::FadeToColor { .. } => {
                let half_duration = self.transition.duration / 2.0;

                match self.phase {
                    TransitionPhase::FadingOut => {
                        let raw_progress = (elapsed / half_duration).clamp(0.0, 1.0);
                        self.progress = self.transition.easing.apply(raw_progress);

                        if raw_progress >= 1.0 {
                            self.phase = TransitionPhase::Midpoint;
                        }
                        false
                    }
                    TransitionPhase::Midpoint => {
                        // Scene swap happens here, then continue to fade in
                        self.phase = TransitionPhase::FadingIn;
                        self.progress = 0.0;
                        false
                    }
                    TransitionPhase::FadingIn => {
                        let fade_in_elapsed = elapsed - half_duration;
                        let raw_progress = (fade_in_elapsed / half_duration).clamp(0.0, 1.0);
                        self.progress = self.transition.easing.apply(raw_progress);

                        raw_progress >= 1.0
                    }
                    TransitionPhase::Crossfading => unreachable!(),
                }
            }
            TransitionKind::Crossfade => {
                let raw_progress = (elapsed / self.transition.duration).clamp(0.0, 1.0);
                self.progress = self.transition.easing.apply(raw_progress);
                self.phase = TransitionPhase::Crossfading;

                raw_progress >= 1.0
            }
        }
    }

    /// Get the fade overlay alpha for fade-to-color transitions.
    ///
    /// Returns (scene_alpha, overlay_alpha) where:
    /// - scene_alpha is how visible the scene should be
    /// - overlay_alpha is how visible the fade color should be
    pub fn get_fade_alpha(&self) -> (f32, f32) {
        match self.phase {
            TransitionPhase::FadingOut => {
                // Scene fades out: 1.0 -> 0.0, overlay fades in: 0.0 -> 1.0
                (1.0 - self.progress, self.progress)
            }
            TransitionPhase::Midpoint => {
                // Fully faded to color
                (0.0, 1.0)
            }
            TransitionPhase::FadingIn => {
                // Scene fades in: 0.0 -> 1.0, overlay fades out: 1.0 -> 0.0
                (self.progress, 1.0 - self.progress)
            }
            TransitionPhase::Crossfading => {
                // Not used for fade-to-color
                (1.0, 0.0)
            }
        }
    }

    /// Get the crossfade blend factor.
    ///
    /// Returns how much of the new scene to show (0.0 = all old, 1.0 = all new).
    pub fn get_crossfade_blend(&self) -> f32 {
        self.progress
    }

    /// Check if we're at the midpoint (time to swap scenes).
    pub fn is_midpoint(&self) -> bool {
        matches!(self.phase, TransitionPhase::Midpoint)
    }

    /// Check if this is a crossfade transition.
    pub fn is_crossfade(&self) -> bool {
        matches!(self.transition.kind, TransitionKind::Crossfade)
    }

    /// Get the fade color if this is a fade-to-color transition.
    pub fn fade_color(&self) -> Option<Color> {
        match &self.transition.kind {
            TransitionKind::FadeToColor { color } => Some(*color),
            _ => None,
        }
    }
}
