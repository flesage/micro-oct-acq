#include <iostream>
#include <QtWidgets>
#include <QtNetwork>
#include <QDir>
#include "config.h"

#include "octserver.h"

OCTServer::OCTServer(QWidget *parent)
    : QDialog(parent)
    , statusLabel(new QLabel)
    , tcpServer(new QTcpServer)
{
    statusLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);

    initServer();
    auto quitButton = new QPushButton(tr("Quit"));
    quitButton->setAutoDefault(false);

    // Connections
    connect(quitButton, &QAbstractButton::clicked, this, &QWidget::close);
    connect(tcpServer, &QTcpServer::newConnection, this, &OCTServer::slot_startConnection);
    connect(this, SIGNAL(sig_start_acquisition(int,int,int)), this, SLOT(slot_performScan(int,int,int)));
    connect(this, SIGNAL(sig_end_acquisition()), this, SLOT(slot_endConnection()));

    // Layout
    auto buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(quitButton);
    buttonLayout->addStretch(1);

    QVBoxLayout *mainLayout = nullptr;
    if (QGuiApplication::styleHints()->showIsFullScreen() || QGuiApplication::styleHints()->showIsMaximized()) {
        auto outerVerticalLayout = new QVBoxLayout(this);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
        auto outerHorizontalLayout = new QHBoxLayout;
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        auto groupBox = new QGroupBox(QGuiApplication::applicationDisplayName());
        mainLayout = new QVBoxLayout(groupBox);
        outerHorizontalLayout->addWidget(groupBox);
        outerHorizontalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::MinimumExpanding, QSizePolicy::Ignored));
        outerVerticalLayout->addLayout(outerHorizontalLayout);
        outerVerticalLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Ignored, QSizePolicy::MinimumExpanding));
    } else {
        mainLayout = new QVBoxLayout(this);
    }

    mainLayout->addWidget(statusLabel);
    mainLayout->addLayout(buttonLayout);

    setWindowTitle(QGuiApplication::applicationDisplayName());
}

void OCTServer::initServer()
{
    // Set the server address and port
    QString ipAddress = OCTServer::getHostAddress();
    const QHostAddress address = QHostAddress(ipAddress);
    quint16 ipPort = SERVER_SOCKET_PORT;

    if (!tcpServer->listen(address, ipPort)) {
        QMessageBox::critical(this, tr("OCT Server"),
                              tr("Unable to start the server: %1.")
                              .arg(tcpServer->errorString()));
        close();
        return;
    }

    statusLabel->setText(tr("The server is running on\n\nIP: %1\nport: %2\n\n"
                            "Waiting for connections ...")
                         .arg(ipAddress).arg(tcpServer->serverPort()));

}

void OCTServer::slot_endConnection()
{
    std::cout << "Closing the connection" << std::endl;
    QByteArray response;
    QDataStream out(&response, QIODevice::WriteOnly);
    out.setVersion(QDataStream::Qt_6_6);
    out << "OCT_done";
    clientConnection->write(response);
}

QString OCTServer::getHostAddress()
{
    QString ipAddress;
    const QList<QHostAddress> ipAddressesList = QNetworkInterface::allAddresses();

    // use the first non-localhost IPv4 address
    for (const QHostAddress &entry : ipAddressesList) {
        if (entry != QHostAddress::LocalHost && entry.toIPv4Address()) {
            ipAddress = entry.toString();
            break;
        }
    }
    // if we did not find one, use IPv4 localhost
    if (ipAddress.isEmpty())
        ipAddress = QHostAddress(QHostAddress::LocalHost).toString();

    return ipAddress;
}

void OCTServer::slot_readTilePosition()
{
    std::cout << "Reading tile positions: (x,y,z)=";

    QByteArray total_data, buffer;
    while(1) {
        buffer = clientConnection->read(1024);
        if (buffer.isEmpty()) {
            break;
        }
        total_data.append(buffer);
    }
    QString data = QString(total_data);
    p_tile_x = data.split(" ")[0].toInt();
    p_tile_y = data.split(" ")[1].toInt();
    p_tile_z = data.split(" ")[2].toInt();

    std::cout << "(" << p_tile_x <<"," << p_tile_y << "," << p_tile_z << ")" << std::endl;
    emit sig_start_acquisition(p_tile_x, p_tile_y, p_tile_z);
}

void OCTServer::slot_startConnection()
{
    std::cout << "Received a connection" << std::endl;
    clientConnection = tcpServer->nextPendingConnection();
    connect(clientConnection, &QIODevice::readyRead, this, &OCTServer::slot_readTilePosition);
}

void OCTServer::slot_performScan(int x, int y, int z)
{
    std::cout << "Starting an OCT acquisition" << std::endl;
    QThread::msleep(5000);
    std::cout << "Ending the OCT acquisition" << std::endl;
    emit sig_end_acquisition();
}
