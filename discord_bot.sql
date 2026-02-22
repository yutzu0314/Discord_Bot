-- discord_bot.sql (baseline + split tables)
-- MariaDB / MySQL, utf8mb4
SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

-- 建議：確保資料庫存在（若你已建立可略過）
-- CREATE DATABASE IF NOT EXISTS `discord_bot`
--   DEFAULT CHARACTER SET utf8mb4
--   COLLATE utf8mb4_uca1400_ai_ci;
-- USE `discord_bot`;

-- --------------------------------------------------------
-- Table: cameras (沿用你現有的)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS `cameras` (
  `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT,
  `guild_id` bigint(20) NOT NULL,
  `channel_id` bigint(20) NOT NULL,
  `name` varchar(100) NOT NULL,
  `name_en` varchar(100) DEFAULT NULL,
  `road_name` varchar(100) DEFAULT NULL,
  `latitude` decimal(9,6) DEFAULT NULL,
  `longitude` decimal(9,6) DEFAULT NULL,
  `stream_url` text DEFAULT NULL,
  `is_active` tinyint(1) NOT NULL DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_cameras_guild_channel` (`guild_id`,`channel_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

-- --------------------------------------------------------
-- (可選保留) Table: reports (舊表，先不刪避免舊程式/資料壞掉)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS `reports` (
  `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `guild_id` bigint(20) DEFAULT NULL,
  `channel_id` bigint(20) DEFAULT NULL,
  `message_id` bigint(20) DEFAULT NULL,
  `reporter_id` bigint(20) DEFAULT NULL,
  `camera_id` bigint(20) UNSIGNED DEFAULT NULL,
  `road_name` varchar(100) DEFAULT NULL,
  `latitude` decimal(9,6) DEFAULT NULL,
  `longitude` decimal(9,6) DEFAULT NULL,
  `image_url` text DEFAULT NULL,
  `category` varchar(50) DEFAULT NULL,
  `note` text DEFAULT NULL,
  `status` enum('pending','approved','rejected') DEFAULT 'pending',
  PRIMARY KEY (`id`),
  KEY `idx_reports_created_at` (`created_at`),
  KEY `idx_reports_channel` (`channel_id`),
  KEY `fk_reports_camera` (`camera_id`),
  CONSTRAINT `fk_reports_camera` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

-- --------------------------------------------------------
-- NEW Table: road_requests (道路/監視器申請與審核)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS `road_requests` (
  `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),

  `guild_id` bigint(20) DEFAULT NULL,
  `admin_channel_id` bigint(20) DEFAULT NULL,
  `message_id` bigint(20) DEFAULT NULL,     -- 管理員頻道那則審核訊息的 message id
  `reporter_id` bigint(20) NOT NULL,        -- 申請者

  `road_name` varchar(100) NOT NULL,
  `image_url` text DEFAULT NULL,            -- 監視器網址/圖片網址
  `note` text DEFAULT NULL,

  -- 狀態：pending / approved / rejected / need_edit
  `status` enum('pending','approved','rejected','need_edit') NOT NULL DEFAULT 'pending',

  `reviewed_by` bigint(20) DEFAULT NULL,
  `reviewed_at` timestamp NULL DEFAULT NULL,

  `camera_id` bigint(20) UNSIGNED DEFAULT NULL,  -- 核准後可綁定 cameras.id

  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_road_requests_message_id` (`message_id`),
  KEY `idx_road_requests_status` (`status`),
  KEY `idx_road_requests_created_at` (`created_at`),
  KEY `idx_road_requests_reporter` (`reporter_id`),
  KEY `fk_road_requests_camera` (`camera_id`),
  CONSTRAINT `fk_road_requests_camera` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

-- --------------------------------------------------------
-- NEW Table: violations (違規事件，用於週報統計)
-- --------------------------------------------------------
CREATE TABLE IF NOT EXISTS `violations` (
  `id` bigint(20) UNSIGNED NOT NULL AUTO_INCREMENT,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),

  `guild_id` bigint(20) DEFAULT NULL,
  `channel_id` bigint(20) DEFAULT NULL,

  `camera_id` bigint(20) UNSIGNED DEFAULT NULL,

  `category` varchar(50) DEFAULT NULL,      -- bike / oloo / ...
  `confidence` float DEFAULT NULL,

  `image_url` text DEFAULT NULL,
  `note` text DEFAULT NULL,

  PRIMARY KEY (`id`),
  KEY `idx_violations_created_at` (`created_at`),
  KEY `idx_violations_channel` (`channel_id`),
  KEY `idx_violations_category` (`category`),
  KEY `fk_violations_camera` (`camera_id`),
  CONSTRAINT `fk_violations_camera` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

COMMIT;